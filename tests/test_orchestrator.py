import importlib.util
import os
import unittest
from unittest import mock

os.environ["MOCK_LLM_RESPONSES"] = "true"  # tests never call real LLM endpoints

from app.graph import USING_LANGGRAPH, initial_state, orchestrator  # noqa: E402

HAS_CRYPTOGRAPHY = importlib.util.find_spec("cryptography") is not None
HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None


def run_graph(prompt, force_local=False):
    return orchestrator.invoke(initial_state(prompt, force_local))


class TestRouting(unittest.TestCase):
    def test_japanese_clinical_prompt_routes_to_local(self):
        result = run_graph("患者のカルテと明日の予約枠を確認してください")
        self.assertTrue(result["pii_detected"])
        self.assertEqual(result["route"], "local_llm")

    def test_english_medical_prompt_routes_to_local(self):
        result = run_graph("Please pull the patient's medical record.")
        self.assertTrue(result["pii_detected"])
        self.assertEqual(result["route"], "local_llm")

    def test_phone_number_inside_japanese_text_routes_to_local(self):
        result = run_graph("連絡先は090-1234-5678です。")
        self.assertTrue(result["pii_detected"])
        self.assertEqual(result["route"], "local_llm")

    def test_general_prompt_routes_to_commercial(self):
        result = run_graph("Explain Big O notation for hash map lookups.")
        self.assertFalse(result["pii_detected"])
        self.assertEqual(result["route"], "commercial_llm")

    def test_force_local_overrides_routing(self):
        result = run_graph("Summarize Python list comprehensions.", force_local=True)
        self.assertFalse(result["pii_detected"])
        self.assertEqual(result["route"], "local_llm")

    def test_mock_mode_makes_no_network_calls(self):
        fake_httpx = mock.Mock()
        with mock.patch("app.graph.httpx", fake_httpx):
            run_graph("患者のカルテを確認してください")
            run_graph("Explain Big O notation for hash map lookups.")
        fake_httpx.Client.assert_not_called()


@unittest.skipUnless(USING_LANGGRAPH, "langgraph is not installed")
class TestGraphWiring(unittest.TestCase):
    def test_classify_runs_first_and_branches_to_both_llm_nodes(self):
        graph = orchestrator.get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("__start__", "classify"), edges)
        self.assertIn(("classify", "local_llm"), edges)
        self.assertIn(("classify", "commercial_llm"), edges)


@unittest.skipUnless(HAS_CRYPTOGRAPHY, "cryptography is not installed")
class TestAes256Cbc(unittest.TestCase):
    def test_round_trip_uses_a_fresh_iv_every_time(self):
        from app.crypto import decrypt_aes256_cbc, encrypt_aes256_cbc

        key = os.urandom(32)
        text = "明日10時の予約を確認しました。"
        ct1, iv1 = encrypt_aes256_cbc(text, key)
        ct2, iv2 = encrypt_aes256_cbc(text, key)
        self.assertNotEqual(iv1, iv2)
        self.assertNotEqual(ct1, ct2)
        self.assertEqual(decrypt_aes256_cbc(ct1, iv1, key), text)
        self.assertEqual(decrypt_aes256_cbc(ct2, iv2, key), text)

    def test_refuses_to_encrypt_without_a_shared_secret(self):
        from app.crypto import EncryptionNotConfiguredError, encrypt_aes256_cbc

        with mock.patch.dict(os.environ, {"AES_SHARED_SECRET": ""}):
            with self.assertRaises(EncryptionNotConfiguredError):
                encrypt_aes256_cbc("hello")


@unittest.skipUnless(HAS_FASTAPI and HAS_CRYPTOGRAPHY, "fastapi/httpx/cryptography are not installed")
class TestApi(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from app.main import app

        self.client = TestClient(app)

    def test_encrypted_response_hides_the_plaintext_answer(self):
        from app.crypto import decrypt_aes256_cbc, get_aes_key

        with mock.patch.dict(os.environ, {"AES_SHARED_SECRET": "unit-test-secret"}):
            resp = self.client.post(
                "/v1/orchestrate",
                json={"prompt": "患者のカルテを確認してください", "encrypt_response": True},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["route"], "local_llm")
            self.assertIsNone(body["answer"])
            payload = body["encrypted_payload"]
            plaintext = decrypt_aes256_cbc(payload["ciphertext_b64"], payload["iv_b64"], get_aes_key())
        self.assertTrue(plaintext.startswith("[MOCK LOCAL:"))

    def test_encryption_without_a_secret_is_rejected_before_any_llm_call(self):
        with mock.patch.dict(os.environ, {"AES_SHARED_SECRET": ""}), mock.patch(
            "app.main.orchestrator"
        ) as fake_graph, self.assertLogs("hybrid_llm_orchestrator", level="ERROR"):
            resp = self.client.post(
                "/v1/orchestrate",
                json={"prompt": "Explain Big O notation.", "encrypt_response": True},
            )
        self.assertEqual(resp.status_code, 500)
        fake_graph.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
