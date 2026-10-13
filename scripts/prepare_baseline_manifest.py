"""Keep official train/dev entries from a previously validated subset manifest."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from voicefake.manifest import REQUIRED_COLUMNS, validate_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
source, output = Path(args.source), Path(args.output)
if output.exists():
    parser.error('Output already exists; choose a new filename')
validate_manifest(source)
with source.open(newline='', encoding='utf-8-sig') as stream:
    rows = [{name: row[name].strip() for name in REQUIRED_COLUMNS}
            for row in csv.DictReader(stream) if row['split'].strip() in {'train', 'dev'}]
if set(row['split'] for row in rows) != {'train', 'dev'}:
    parser.error('Source must contain both train and dev')
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('x', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
print(json.dumps({'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                  'audit': validate_manifest(output)}, indent=2))
