import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from temporal_ssl.cli import load_config
from temporal_ssl.eeg_data import EEGDataset, aggregate_participants, fit_normalizer, subject_splits
from temporal_ssl.eeg_models import FirstHalfPrediction
from temporal_ssl.eeg_pilot import half_baselines, mask_runs, run_pilot
from temporal_ssl.models import TemporalSSL, patchify


def test_eeg_fixed_half_targets_excluded_and_capacity():
    torch.manual_seed(3)
    model = FirstHalfPrediction(3, 40, patch_size=5, d_model=8, nhead=2, layers=2).eval()
    x = torch.randn(2,40,3)
    changed = x.clone()
    changed[:,20:] = 1000
    result = model(x)
    torch.testing.assert_close(result["prediction"],model(changed)["prediction"])
    torch.testing.assert_close(result["target"],patchify(x,5)[:,4:])
    assert result["prediction"].shape == (2,4,15)
    torch.testing.assert_close(result["loss"],(result["prediction"]-result["target"]).square().mean())
    torch.testing.assert_close(model.representations(x)["sequence"],model.representations(changed)["sequence"])
    local=model.representations(x)["local"]
    early=x.clone();early[:,10:20]+=100
    torch.testing.assert_close(local[:,:2],model.representations(early)["local"][:,:2])
    masked=TemporalSSL(3,40,objective="masked",patch_size=5,d_model=8,nhead=2,layers=2)
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in masked.parameters())
    mask=torch.zeros(2,8,dtype=torch.bool);mask[:,2:4]=True
    altered=x.clone().reshape(2,8,5,3);altered[mask]=10000
    masked.eval()
    torch.testing.assert_close(masked(x,mask)["prediction"],masked(altered.reshape_as(x),mask)["prediction"])


def make_people(tmp_path, n_per_class=10, length=40, channels=3):
    rng=np.random.default_rng(600)
    rows=[]
    for group in ["AD","FTD","HC"]:
        for i in range(n_per_class):
            pid=f"{group}-{i:02}"
            path=tmp_path/f"{pid}.npz"
            np.savez(path,x=rng.normal(size=(2,length,channels)).astype(np.float32),
                     spectral=rng.normal(size=(2,8)),source_start=[0,length])
            rows.append({"participant_id":pid,"group":group,"cache":str(path),"excluded":False,
                         "sampling_hz":5.0})
    return rows


def test_eeg_disjoint_normalization_shapes_and_aggregation(tmp_path):
    rows=make_people(tmp_path)
    split=subject_splits(rows)
    assert split==subject_splits(rows)
    sets=[{r["participant_id"] for r in part} for part in split.values()]
    assert len(set.union(*sets))==30
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    ds={k:EEGDataset(v) for k,v in split.items()}
    norm=fit_normalizer(ds["train"])
    np.testing.assert_allclose(norm.mean,ds["train"].x.mean(axis=(0,1)),atol=1e-7)
    np.testing.assert_allclose(norm.std,ds["train"].x.std(axis=(0,1)),atol=1e-6)
    original=norm.mean.copy()
    ds["test"].x+=100
    np.testing.assert_array_equal(original,fit_normalizer(ds["train"]).mean)
    ds["train"].normalizer=norm
    record=ds["train"][0]
    assert record["x"].shape==(40,3) and record["latents"] is None
    assert patchify(record["x"][None],5).shape==(1,8,15)
    metadata=[{"participant_id":"a","group":"AD"},{"participant_id":"a","group":"AD"},
              {"participant_id":"b","group":"HC"}]
    ids,x,y=aggregate_participants(np.array([[1.,3],[3,5],[8,10]]),metadata)
    assert ids==["a","b"] and y.tolist()==["AD","HC"]
    np.testing.assert_array_equal(x,[[2,4],[8,10]])
    metadata[1]["group"]="FTD"
    with pytest.raises(ValueError): aggregate_participants(np.ones((3,2)),metadata)


def test_eeg_baselines_and_mask_lengths():
    x=np.arange(40,dtype=float)[:,None]
    result=half_baselines(x)
    np.testing.assert_allclose(result["linear_trend"],x[20:])
    changed=x.copy();changed[20:]+=1000
    for key,value in result.items():np.testing.assert_array_equal(value,half_baselines(changed)[key])
    np.testing.assert_array_equal(mask_runs([True,True,False,True,False,True,True,True]),[2,1,3])


def test_eeg_complete_smoke(tmp_path):
    people=make_people(tmp_path)
    cache=tmp_path/"cache";cache.mkdir()
    (cache/"manifest.json").write_text(json.dumps({"identity":"fixture", "channels":["a","b","c"],
             "specification":{"max_epochs":2},"participants":people}))
    cfg=load_config("configs/eeg_pilot.json")
    cfg.update(cache_dir=str(cache),max_epochs_per_participant=2)
    cfg["model"].update(patch_size=5,d_model=8,nhead=2,layers=1,context=4)
    cfg["spans"]={"short":1,"medium":2,"long":3}
    cfg["training"].update(device="cpu",epochs=1,batch_size=4)
    result=run_pilot(cfg,tmp_path/"out",smoke=True)
    assert len(result["models"])==4 and len(result["classification"])==7
    assert len({r["parameters"] for r in result["models"].values()})==1
    for name in result["models"]:
        assert np.isfinite(result["models"][name]["reconstruction"]["neural"]["mean"])
        assert (tmp_path/"out"/name/"classifier.joblib").exists()
    assert (tmp_path/"out/report.md").exists()
