"""A freeze detects scientific-input drift; it is not a cryptographic signature."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from voicefake.audio import _file_hash
from voicefake.freeze import create_freeze, seal_specification, verify_freeze


class FreezeTests(unittest.TestCase):
    def fixture(self,root):
        (root/"code.py").write_text("fixed recipe",encoding="utf-8")
        (root/"model.joblib").write_bytes(b"trusted-fixture-model")
        body={"schema_version":1,"status":"frozen","primary_model":"svm",
              "models":{"svm":{"decision_rule":{"mode":"score_at_least","threshold":.25}}},
              "code_config_evidence":{"code.py":_file_hash(root/"code.py")},
              "local_artifacts":{"model.joblib":_file_hash(root/"model.joblib")},
              "codec":{"fixture":True},"audio_runtime":{"fixture":True},"model_runtime":{}}
        path=root/"lock.json";path.write_text(json.dumps(seal_specification(body)),encoding="utf-8")
        return path,body

    def mocked_runtime(self):
        stack=ExitStack()
        stack.enter_context(patch("voicefake.freeze.codec_identity",return_value={"fixture":True}))
        stack.enter_context(patch("voicefake.freeze.runtime_identity",return_value={"fixture":True}))
        return stack

    def test_seal_is_canonical_and_verifier_only_reads_model_bytes(self):
        with tempfile.TemporaryDirectory() as directory,self.mocked_runtime():
            root=Path(directory);path,body=self.fixture(root)
            self.assertEqual(seal_specification(body),seal_specification(dict(reversed(list(body.items())))))
            result=verify_freeze(path,root)
            self.assertEqual(result["status"],"verified")
            self.assertEqual(result["file_bindings_verified"],2)
            self.assertFalse(result["evaluation_audio_processed"])
            with self.assertRaisesRegex(ValueError,"reseal"):
                seal_specification(seal_specification(body))

    def test_threshold_edit_fails_the_content_lock(self):
        with tempfile.TemporaryDirectory() as directory,self.mocked_runtime():
            root=Path(directory);path,_=self.fixture(root)
            spec=json.loads(path.read_text(encoding="utf-8"));spec["models"]["svm"]["decision_rule"]["threshold"]=.5
            path.write_text(json.dumps(spec),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"content checksum"):
                verify_freeze(path,root)

    def test_code_or_model_drift_is_detected(self):
        for name in ("code.py","model.joblib"):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory,self.mocked_runtime():
                root=Path(directory);path,_=self.fixture(root)
                (root/name).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError,"binding changed"):
                    verify_freeze(path,root)

    def test_relative_paths_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as directory,self.mocked_runtime():
            root=Path(directory);path,body=self.fixture(root)
            body["local_artifacts"]={"../outside":"0"*64}
            path.write_text(json.dumps(seal_specification(body)),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"repository-relative"):
                verify_freeze(path,root)

    def test_runtime_change_and_overwriting_are_refused(self):
        with tempfile.TemporaryDirectory() as directory,self.mocked_runtime():
            root=Path(directory);path,_=self.fixture(root)
            with patch("voicefake.freeze.codec_identity",return_value={"different":True}):
                with self.assertRaisesRegex(ValueError,"runtime differs"):
                    verify_freeze(path,root)
            with self.assertRaisesRegex(ValueError,"already exists"):
                create_freeze(root,output="lock.json")


if __name__=="__main__":
    unittest.main()
