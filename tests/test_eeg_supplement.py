import numpy as np
import pytest

from temporal_ssl.eeg_supplement import bootstrap_scores, metadata_counts


def test_bootstrap_preserves_people_across_models_and_seeds():
    truth = np.array(['AD','AD','FTD','FTD','HC','HC'])
    # Seed-specific differences deliberately differ; the replicate must average
    # the paired differences, rather than independently resample seeds/models.
    cube = np.array([[[1,0,1,0,0,1],[0,1,0,1,1,0]],
                     [[0,0,0,1,1,0],[1,0,1,0,0,1]]],dtype=float)
    draws, indices = bootstrap_scores(cube, truth, replicates=37, seed=9)
    manual = []
    for replicate in range(37):
        differences = []
        for seed in range(2):
            value = 0
            for sampled in indices.values():
                ix = sampled[replicate]
                value += (cube[0,seed,ix]-cube[1,seed,ix]).mean()/3
            differences.append(value)
        manual.append(np.mean(differences))
    np.testing.assert_allclose((draws[0]-draws[1]).mean(0),manual,atol=1e-15)
    same, _ = bootstrap_scores(np.stack([cube[0],cube[0]]),truth,replicates=37,seed=9)
    np.testing.assert_array_equal(same[0]-same[1],0)
    repeated, again = bootstrap_scores(cube,truth,replicates=37,seed=9)
    np.testing.assert_array_equal(draws,repeated)
    for diagnosis, sampled in indices.items():
        assert np.all(truth[sampled] == diagnosis)
        np.testing.assert_array_equal(sampled,again[diagnosis])
    with pytest.raises(ValueError):
        bootstrap_scores(cube[:,:,:4],truth[:4])


def test_qc_mask_removals_are_separate_from_boundary_tails():
    mask = np.r_[np.ones(5000,bool),np.zeros(100,bool),np.ones(8500,bool)]
    metadata = {'pnts':13500,'srate':500,
                'etc':{'clean_sample_mask':mask},'event':{'type':'boundary','latency':5000}}
    counts = metadata_counts(metadata)
    assert counts['original_samples']==13600
    assert counts['provider_removed_samples']==100
    assert counts['provider_removed_seconds']==.2
    assert counts['internal_boundaries']==1
    assert counts['contiguous_segments']==2
    assert counts['complete_candidates']==3
    assert counts['segment_tail_samples']==1500
    assert counts['segment_tail_seconds']==3
    assert counts['retained_samples']==3*4000+1500
