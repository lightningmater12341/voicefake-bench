"""Timing fixtures check accounting and boundaries; they are not laptop measurements."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from voicefake.costs import CostConfig, hardware_identity, measure_call, pass_summaries, run_costs, summarize_timings


class CostTests(unittest.TestCase):
    def test_timer_wraps_callable_and_preserves_result(self):
        events = []
        ticks = iter([100, 350])
        def clock():
            events.append("clock"); return next(ticks)
        def function():
            events.append("function"); return {"fixture": True}
        result, elapsed = measure_call(function, clock)
        self.assertEqual(result, {"fixture": True})
        self.assertEqual(elapsed, 250)
        self.assertEqual(events, ["clock", "function", "clock"])

    def test_nonpositive_or_noninteger_clock_results_fail(self):
        for ticks in ([10,10],[10,9],[1.,2.]):
            with self.subTest(ticks=ticks), self.assertRaises(ValueError):
                iterator=iter(ticks); measure_call(lambda: None, lambda: next(iterator))

    def test_summary_weights_rows_and_audio_not_average_reciprocals(self):
        rows=[{"elapsed_ns":100,"rows":1,"audio_seconds":1.}, {"elapsed_ns":900,"rows":3,"audio_seconds":3.}]
        summary=summarize_timings(rows)
        self.assertEqual(summary["timed_total_ns"],1000)
        self.assertEqual(summary["processed_rows_with_repeats"],4)
        self.assertAlmostEqual(summary["amortized_ms_per_clip"],.00025)
        self.assertAlmostEqual(summary["clips_per_timed_second"],4_000_000)
        self.assertAlmostEqual(summary["call_latency_ms"]["p50"],.0005)
        self.assertAlmostEqual(summary["call_latency_ms"]["p95"],.00086)
        self.assertAlmostEqual(summary["real_time_factor"],.00000025)

    def test_batch_latency_is_not_one_clip_latency(self):
        summary=summarize_timings([{"elapsed_ns":1_000_000,"rows":700,"audio_seconds":2800.}])
        self.assertEqual(summary["call_latency_ms"]["p50"],1.)
        self.assertAlmostEqual(summary["amortized_ms_per_clip"],1/700)
        self.assertEqual(summary["timed_calls"],1)

    def test_invalid_timing_records_fail_instead_of_publishing(self):
        good={"elapsed_ns":100,"rows":1,"audio_seconds":1.}
        for field,value in (("elapsed_ns",0),("elapsed_ns",1.5),("rows",0),("rows",True),("audio_seconds",np.nan),("audio_seconds",0.)):
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                summarize_timings([{**good,field:value}])
        with self.assertRaises(ValueError): summarize_timings([])

    def test_passes_do_not_mix_models_repeats_or_modes(self):
        base={"stage":"inference","model":"svm","mode":"one_clip","repeat":0,"elapsed_ns":100,"rows":1,"audio_seconds":1.}
        rows=[base,{**base,"elapsed_ns":300},{**base,"repeat":1},{**base,"model":"logistic"},{**base,"mode":"batch_700","rows":700}]
        passes=pass_summaries(rows)
        self.assertEqual(len(passes),4)
        selected=next(p for p in passes if p["model"]=="svm" and p["repeat"]==0 and p["mode"]=="one_clip")
        self.assertEqual(selected["timed_calls"],2)
        self.assertEqual(selected["timed_total_ns"],400)

    def test_config_is_fixed_and_unknown_fields_fail(self):
        for values in ({"thread_limit":2},{"batch_repeats":True},{"schema_version":2},{"warmup_clips":0}):
            with self.assertRaises(ValueError): CostConfig(**values)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"costs.json"
            path.write_text('{"surprise":1}',encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Unknown"): CostConfig.from_json(path)
            path.write_text('[]',encoding="utf-8")
            with self.assertRaises(ValueError): CostConfig.from_json(path)

    def test_hardware_output_is_allowlisted_and_omits_identifiers(self):
        info=hardware_identity()
        self.assertEqual(set(info),{"system","release","os_version","machine","cpu_description","logical_cpus","physical_memory_bytes","power_ac_online"})
        self.assertTrue(info["system"])
        self.assertTrue(info["logical_cpus"] is None or info["logical_cpus"]>0)
        json.dumps(info,allow_nan=False)

    def test_runner_checks_freeze_before_audio_or_models(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("voicefake.costs.verify_freeze",side_effect=ValueError("freeze drift")), patch("voicefake.costs.load_audio") as decode, patch("voicefake.costs.joblib.load") as load:
                with self.assertRaisesRegex(ValueError,"freeze drift"): run_costs(folder)
                decode.assert_not_called();load.assert_not_called()

    def test_runner_refuses_evaluation_metadata_before_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"configs").mkdir();(root/"data/manifests").mkdir(parents=True)
            (root/"configs/costs.json").write_text('{}',encoding="utf-8")
            (root/"configs/frozen-experiment.json").write_text('{}',encoding="utf-8")
            (root/"data/manifests/baseline-train-dev.csv").write_text(
                "audio_path,label,split,speaker_id,attack_id\ntrain/a.flac,bonafide,train,s1,-\neval/b.flac,spoof,eval,s2,A07\n",encoding="utf-8")
            with patch("voicefake.costs.verify_freeze",return_value={}), patch("voicefake.costs.load_audio") as decode, patch("voicefake.costs.joblib.load") as load:
                with self.assertRaisesRegex(ValueError,"evaluation"): run_costs(root)
                decode.assert_not_called();load.assert_not_called()

    def test_runner_refuses_overwrite_or_output_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"artifacts/existing").mkdir(parents=True)
            for output in ("../outside","docs/run","artifacts/existing"):
                with self.subTest(output=output),self.assertRaises(ValueError): run_costs(root,output)


if __name__=="__main__":
    unittest.main()
