"""Prepare the original EEG panels, split, normalization and spectral control, without SSL.

Fresh reproduction only. Refuses to overwrite an existing pilot output directory.
The original, unchanged preprocessing and split functions perform all preparation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_ssl.eeg_data import prepare_cache, aggregate_participants
from temporal_ssl.eeg_pilot import build_datasets, classify
from temporal_ssl.experiment import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/submission/eeg_pilot.json',type=Path)
    parser.add_argument('--output',default='results/eeg_pilot',type=Path)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError(f'Preserve existing artifacts: {args.output} already exists')
    prepare_cache(cfg['data_dir'],cfg['cache_dir'],cfg['max_epochs_per_participant'])
    # All subsequent files are new artifacts. No existing split/scaler is replaced.
    args.output.mkdir(parents=True)
    datasets=build_datasets(cfg,args.output)
    actual=pd.DataFrame([dict(participant_id=r['participant_id'],diagnosis=r['group'],split=k)
        for k,v in json.loads((args.output/'participants.json').read_text()).items() for r in v])
    expected=pd.read_csv('reports/submission_assets/participant_split.csv')
    pd.testing.assert_frame_equal(actual.sort_values('participant_id').reset_index(drop=True),
                                  expected.sort_values('participant_id').reset_index(drop=True))
    np.testing.assert_array_equal([len(datasets[k]) for k in ['train','validation','test']],[1248,432,432])
    save_json(args.output/'config.json',cfg)
    features={k:aggregate_participants(v.spectral,v.metadata) for k,v in datasets.items()}
    scores=classify(features,cfg,args.output/'spectral')
    print('Prepared original split/normalizer; no SSL trained. Spectral test BA:',scores['test']['balanced_accuracy'])


if __name__=='__main__':
    main()
