"""Run the corrected, versioned experiment from any working directory."""
import argparse
import json
from pathlib import Path

from src.recent_features import build_recent_features
from src.upgraded_training import train_upgraded, file_hash

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-features',action='store_true',help='Recompute windows even if source hashes match')
    parser.add_argument('--quick',action='store_true',help='50 trees and one depth for a development smoke run; separate outputs')
    args = parser.parse_args()
    base = ROOT/'data/processed_market_valuations.parquet'
    if not base.exists():
        from src.data_pipeline import build_analysis_dataset, verify_leakage_invariants
        build_analysis_dataset(str(ROOT),str(base))
        verify_leakage_invariants(str(base))
    data = ROOT/'data/upgraded_market_valuations.parquet'
    manifest_path = ROOT/'data/upgraded_features_manifest.json'
    fingerprints = {str(p.relative_to(ROOT)):file_hash(p) for p in (
        base,ROOT/'appearances.csv',ROOT/'src/recent_features.py')}
    prior = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if args.rebuild_features or not data.exists() or fingerprints!=prior.get('source_hashes'):
        print('Building strict 90/180/365-day windows...',flush=True)
        build_recent_features(base,ROOT/'appearances.csv',data)
        manifest_path.write_text(json.dumps({'source_hashes':fingerprints,'windows':'[date-window,date)'},indent=2),encoding='utf-8')
    destination = ROOT/'models'/('upgraded_smoke' if args.quick else 'upgraded')
    metadata = train_upgraded(data,destination,depths=(5,) if args.quick else (5,7),trees=50 if args.quick else 250)
    metadata['source_hashes'] = fingerprints
    metadata['training_code_hashes'] = {str(p.relative_to(ROOT)):file_hash(p) for p in (
        ROOT/'src/preprocessing.py',ROOT/'src/temporal.py',ROOT/'src/upgraded_training.py',Path(__file__))}
    metadata['development_smoke_run'] = args.quick
    (destination/'run_metadata.json').write_text(json.dumps(metadata,indent=2,default=str,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
