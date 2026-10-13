"""Model-boundary tests use controlled fixtures, not speech benchmark results."""

import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

from voicefake.baselines import BaselineConfig, classification_metrics, fit_baselines, load_feature_partitions, run_baselines
from voicefake.feature_cache import extract_manifest
from voicefake.manifest import REQUIRED_COLUMNS


class BaselineTests(unittest.TestCase):
    def arrays(self):
        generator = np.random.default_rng(5)
        train = generator.normal(size=(60, 4))
        dev = generator.normal(size=(60, 4))
        labels = np.repeat([0, 1], 30)
        train[labels == 1, 0] += 2
        dev[labels == 1, 0] += 2
        return train, labels, dev, labels.copy()

    def test_metrics_known_confusion_and_orientation(self):
        result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        self.assertEqual(result['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertEqual(result['accuracy'], 0.75)
        self.assertEqual(result['balanced_accuracy'], 0.75)
        self.assertEqual(result['spoof_recall'], 1)
        self.assertEqual(result['false_positive_rate'], 0.5)
        self.assertAlmostEqual(result['spoof_f1'], 0.8)

    def test_scaler_and_coefficients_ignore_development_shift(self):
        train, labels, dev, dev_labels = self.arrays()
        first, _, report = fit_baselines(train, labels, dev, dev_labels)
        shifted, _, _ = fit_baselines(train, labels, dev + 1000, dev_labels)
        scaler = first['logistic'].named_steps['scaler']
        np.testing.assert_allclose(scaler.mean_, train.mean(axis=0), atol=1e-12)
        self.assertEqual(report['scaler_fit_rows'], 60)
        np.testing.assert_array_equal(first['logistic'].named_steps['classifier'].coef_,
                                      shifted['logistic'].named_steps['classifier'].coef_)
        np.testing.assert_array_equal(first['logistic'].named_steps['classifier'].intercept_,
                                      shifted['logistic'].named_steps['classifier'].intercept_)

    def test_majority_baseline_ignores_features(self):
        train, labels, dev, dev_labels = self.arrays()
        labels[:10] = 1
        models, scores, _ = fit_baselines(train, labels, dev, dev_labels)
        self.assertTrue(np.all(scores['majority']['dev'] == 1))
        np.testing.assert_array_equal(models['majority'].predict(dev),
                                      models['majority'].predict(dev * 1000))

    def test_insufficient_or_nonfinite_features_rejected(self):
        train, labels, dev, dev_labels = self.arrays()
        with self.assertRaisesRegex(ValueError, 'at least'):
            fit_baselines(train[:5], labels[:5], dev, dev_labels)
        train[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'invalid'):
            fit_baselines(train, labels, dev, dev_labels)

    def test_recipe_and_invalid_configuration(self):
        path = Path(__file__).resolve().parents[1] / 'configs/baseline.json'
        self.assertEqual(asdict(BaselineConfig.from_json(path)), asdict(BaselineConfig()))
        for config in ({'C': 0}, {'seed': -1}, {'max_iter': 0}, {'min_class_count': 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                BaselineConfig(**config)

    def dataset(self, root):
        rows = []
        for split in ('train', 'dev'):
            for i in range(40):
                file = f'{split}/{i}.wav'
                path = root / file
                path.parent.mkdir(exist_ok=True)
                frequency = 300 + i * 30 + (7 if split == 'dev' else 0)
                signal = (0.2 * np.sin(2 * np.pi * frequency * np.arange(8000) / 16000)).astype(np.float32)
                sf.write(path, signal, 16000, subtype='PCM_16')
                rows.append({'audio_path': file, 'label': 'bonafide' if i < 20 else 'spoof',
                             'split': split, 'speaker_id': split + '_speaker',
                             'attack_id': '-' if i < 20 else 'fixture'})
        manifest = root / 'manifest.csv'
        with manifest.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader(); writer.writerows(rows)
        index_path = root / 'index.json'
        index = extract_manifest(manifest, root, root / 'cache', index_path)
        return manifest, index_path, index

    def test_local_artifact_reload_matches_recorded_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, index, _ = self.dataset(root)
            report = run_baselines(index, manifest, root / 'cache', root / 'run')
            x, _, _, _ = load_feature_partitions(index, manifest, root / 'cache')
            model = joblib.load(root / 'run/logistic.joblib')
            with np.load(root / 'run/development_predictions.npz', allow_pickle=False) as stored:
                np.testing.assert_allclose(model.predict_proba(x['dev'])[:, 1], stored['logistic_scores'], atol=1e-12)
            self.assertFalse(report['development_used_for_fit'])
            self.assertEqual(report['scaler_fit_rows'], 40)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run_baselines(index, manifest, root / 'cache', root / 'run')

    def test_label_membership_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, index_path, index = self.dataset(root)
            index['records'][0]['label'] = 'spoof'
            index_path.write_text(json.dumps(index), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'membership'):
                load_feature_partitions(index_path, manifest, root / 'cache')

    def test_evaluation_membership_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, index_path, index = self.dataset(root)
            for row in index['records']:
                if row['split'] == 'train':
                    row['split'] = 'eval'
            with manifest.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerows([{name: row[name] for name in REQUIRED_COLUMNS} for row in index['records']])
            import hashlib
            index['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            index_path.write_text(json.dumps(index), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'only train and dev'):
                load_feature_partitions(index_path, manifest, root / 'cache')

    def test_corrupt_feature_file_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, index_path, index = self.dataset(root)
            vector = root / 'cache' / index['records'][0]['feature_file']
            np.save(vector, np.zeros(40, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_feature_partitions(index_path, manifest, root / 'cache')


if __name__ == '__main__':
    unittest.main()
