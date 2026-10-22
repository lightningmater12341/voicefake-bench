"""Evaluation boundaries use explicit synthetic metadata/scores, never reported speech metrics."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from voicefake.evaluation import _claim_run, _evaluation_rows, _save_arrays, _write_json, audit_evaluation, run_evaluation
from voicefake.freeze import MODELS
from voicefake.robustness import CONDITIONS, summarize_paired


class EvaluationTests(unittest.TestCase):
    def test_eval_summary_requires_explicit_partition_and_keeps_pairing(self):
        rows=[{"audio_path":f"eval/{i}.flac","split":"eval","speaker_id":speaker,
               "label":label,"attack_id":"-" if label=="bonafide" else "A07"}
              for i,(speaker,label) in enumerate([("a","bonafide"),("a","spoof"),("b","bonafide"),("b","spoof")])]
        scores={"fixed":{c:np.array([0.,1.,0.,1.]) for c in CONDITIONS}}
        definitions={"fixed":{"decision_rule":{"mode":"score_at_least","threshold":.5}}}
        plan={"method":"paired speaker-block bootstrap","replicates":20,"seed":42,"confidence":.95,
              "interval":"percentile; numpy quantile method=linear","threshold_reselection":False,"invalid_draws":"never replace"}
        with self.assertRaises(ValueError):summarize_paired(rows,scores,definitions,plan)
        result=summarize_paired(rows,scores,definitions,plan,partition="eval")
        self.assertEqual(result["bootstrap"]["valid_draws"],20)
        self.assertEqual(result["models"]["fixed"]["mp3_32kbps"]["statistics"]["roc_auc"]["paired_difference_interval_95"],[0.,0.])
        rows[0]["split"]="dev"
        with self.assertRaises(ValueError):summarize_paired(rows,scores,definitions,plan,partition="eval")
        with self.assertRaises(ValueError):summarize_paired(rows,scores,definitions,plan,partition="train")

    def manifest(self,root):
        path=root/"data/eval.csv";path.parent.mkdir()
        rows=[{"audio_path":f"eval/g{i}.flac","label":"bonafide","split":"eval","speaker_id":"s1","attack_id":"-"} for i in range(100)]
        rows.extend({"audio_path":f"eval/A{attack}_{i}.flac","label":"spoof","split":"eval","speaker_id":"s2","attack_id":f"A{attack:02d}"} for attack in range(7,20) for i in range(100))
        self.write_rows(path,rows)
        return path,rows,{"dataset":{"evaluation_manifest":"data/eval.csv"}}

    def write_rows(self,path,rows):
        with path.open("w",newline="",encoding="utf-8") as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def test_frozen_evaluation_count_and_attack_membership_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,rows,spec=self.manifest(root)
            self.assertEqual(len(_evaluation_rows(root,spec)[1]),1400)
            rows[-1]["attack_id"]="A18";self.write_rows(path,rows)
            with self.assertRaisesRegex(ValueError,"original frozen"):_evaluation_rows(root,spec)

    def test_evaluation_metadata_rejects_wrong_split_and_extra_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,rows,spec=self.manifest(root)
            rows[0]["split"]="dev";self.write_rows(path,rows)
            with self.assertRaises(ValueError):_evaluation_rows(root,spec)
            rows[0]["split"]="eval";rows.append({**rows[0],"audio_path":"eval/extra.flac"});self.write_rows(path,rows)
            with self.assertRaises(ValueError):_evaluation_rows(root,spec)

    def test_only_one_claim_per_freeze_and_resume_needs_exact_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/"artifacts/evaluation/run";identity={"frozen_lock_sha256":"a"*64,"code":"v1"}
            claim,state=_claim_run(root,target,identity,False)
            self.assertTrue(target.is_dir());self.assertEqual(state["status"],"started")
            with self.assertRaisesRegex(ValueError,"resume"):_claim_run(root,target,identity,False)
            self.assertEqual(_claim_run(root,target,identity,True)[0],claim)
            with self.assertRaisesRegex(ValueError,"already claimed"):_claim_run(root,target,{**identity,"code":"v2"},True)
            with self.assertRaisesRegex(ValueError,"already claimed"):_claim_run(root,root/"artifacts/evaluation/other",identity,False)

    def test_complete_claim_cannot_run_again_even_with_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/"artifacts/evaluation/run";identity={"frozen_lock_sha256":"b"*64}
            claim,state=_claim_run(root,target,identity,False);state["status"]="complete";_write_json(claim,state)
            for resume in (False,True):
                with self.assertRaisesRegex(ValueError,"already complete"):_claim_run(root,target,identity,resume)

    def test_resume_cannot_adopt_unclaimed_or_missing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/"artifacts/evaluation/run";identity={"frozen_lock_sha256":"c"*64}
            with self.assertRaises(ValueError):_claim_run(root,target,identity,True)
            claim,state=_claim_run(root,target,identity,False);target.rmdir()
            with self.assertRaisesRegex(ValueError,"output is missing"):_claim_run(root,target,identity,True)

    def test_atomic_evidence_preserves_values_and_rejects_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/"journal.json";_write_json(path,{"status":"valid"})
            with self.assertRaises(ValueError):_write_json(path,{"score":float("nan")})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),{"status":"valid"})
            arrays=root/"scores.npz";_save_arrays(arrays,{"scores":np.array([.25,.75])})
            with np.load(arrays,allow_pickle=False) as saved:np.testing.assert_array_equal(saved["scores"],[.25,.75])
            self.assertFalse(any(p.suffix==".tmp" for p in root.iterdir()))

    def test_freeze_verified_before_evaluation_decode_or_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            for runner in (audit_evaluation,run_evaluation):
                with patch("voicefake.evaluation.verify_freeze",side_effect=ValueError("freeze drift")),patch("voicefake.evaluation.load_audio") as decode,patch("voicefake.evaluation.fingerprint_recording") as fingerprint,patch("voicefake.evaluation.joblib.load") as load:
                    with self.assertRaisesRegex(ValueError,"freeze drift"):runner(folder)
                    decode.assert_not_called();fingerprint.assert_not_called();load.assert_not_called()

    def test_failed_content_audit_blocks_all_scoring_and_claims(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"configs").mkdir();(root/"artifacts/evaluation-audit/day13").mkdir(parents=True)
            spec={"conditions":list(CONDITIONS),"compared_models":list(MODELS)}
            _write_json(root/"configs/frozen-experiment.json",spec)
            _write_json(root/"artifacts/evaluation-audit/day13/report.json",{"status":"review_required"})
            with patch("voicefake.evaluation.verify_freeze",return_value={}),patch("voicefake.evaluation._evaluation_rows",return_value=(root/"metadata.csv",[{}])),patch("voicefake.evaluation.joblib.load") as load,patch("voicefake.evaluation._claim_run") as claim:
                with self.assertRaisesRegex(ValueError,"passing evaluation content audit"):run_evaluation(root)
                load.assert_not_called();claim.assert_not_called()

    def test_output_paths_cannot_escape_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            for runner in (audit_evaluation,run_evaluation):
                for output in ("../outside","docs/run"):
                    with self.assertRaises(ValueError):runner(folder,output)


if __name__=="__main__":unittest.main()
