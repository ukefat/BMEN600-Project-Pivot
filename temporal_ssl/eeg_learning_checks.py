"""Training-only EEG memorization and visible-input sensitivity checks; no model tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from .baselines import fill_masked
from .eeg_pilot import make_eeg_model
from .experiment import save_json, seed_everything
from .models import patchify, span_mask
from .plots import plt, finish


def select_training_windows(root, count=4):
    partitions = json.loads((root/"participants.json").read_text())
    train = partitions["train"]
    selected = []
    for group in ("AD", "FTD", "HC"):
        selected.append(next(person for person in train if person["group"] == group))
    selected += [person for person in train if person not in selected]
    selected = selected[:count]
    held_out = {p["participant_id"] for split in ("validation", "test") for p in partitions[split]}
    assert not held_out & {p["participant_id"] for p in selected}
    with np.load(root/"normalization.npz") as stats:
        mean, std = stats["mean"], stats["std"]
    windows, metadata = [], []
    for person in selected:
        with np.load(person["cache"]) as cached:
            windows.append((cached["x"][0]-mean)/std)
            metadata.append({"participant_id": person["participant_id"], "group": person["group"],
                             "cache": person["cache"], "cached_epoch_index": 0,
                             "source_start_sample": int(cached["source_start"][0])})
    return torch.from_numpy(np.stack(windows)), metadata


def reconstruction_score(prediction, target, mask):
    return float((prediction-target).square().mean(-1)[mask].mean())


def baseline_scores(x, mask, p):
    patches = patchify(x, p).numpy()
    scores = {"training_mean": float(np.mean(patches[mask.numpy()]**2))}
    for method in ("linear", "nearest"):
        predictions = np.stack([fill_masked(row.reshape(-1,p,x.shape[-1]), hidden.numpy(), method).reshape(row.shape)
                                for row,hidden in zip(patches,mask)])
        scores[method] = float(np.mean((predictions[mask.numpy()]-patches[mask.numpy()])**2))
    return scores


@torch.no_grad()
def input_checks(model, x, mask, donor, seed):
    model.eval()
    p = model.patch_size
    target = patchify(x, p)
    original = model(x, mask)["prediction"]
    patches = patchify(x, p)
    zero = patches.clone().masked_fill((~mask)[...,None],0)
    shuffled = patches.clone()
    generator = torch.Generator().manual_seed(seed)
    for row in range(len(x)):
        indices = (~mask[row]).nonzero().flatten()
        permutation = torch.randperm(len(indices), generator=generator).to(x.device)
        shuffled[row,indices] = patches[row,indices[permutation]]
    other = torch.where(mask[...,None], patches, patchify(donor,p))
    scores = {}
    for name, changed in (("original",patches), ("zero_visible",zero),
                          ("shuffle_visible",shuffled), ("other_window_visible",other)):
        prediction = original if name == "original" else model(changed.reshape_as(x),mask)["prediction"]
        scores[name] = {"mse": reconstruction_score(prediction,target,mask),
                       "prediction_change_rms": float((prediction-original)[mask].square().mean().sqrt())}
    hidden_changed = patches.clone()
    hidden_changed[mask] += 1000
    hidden_prediction = model(hidden_changed.reshape_as(x),mask)["prediction"]
    scores["masked_value_perturbation_max_change"] = float((hidden_prediction-original).abs().max())
    return scores


def gradient_checks(model, x, mask):
    model.eval()
    source = x.detach().clone().requires_grad_(True)
    predictions = model(source,mask)["prediction"][mask]
    gradient, = torch.autograd.grad(predictions.square().mean(),source)
    gradient = patchify(gradient,model.patch_size)
    return {"visible_input_gradient_l2": float(gradient[~mask].norm()),
            "masked_input_gradient_l2": float(gradient[mask].norm())}


def run_checks(pilot, output, steps=2000, device="cuda"):
    if steps < 1:
        raise ValueError("steps must be positive")
    pilot, output = Path(pilot), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"summary.json").exists():
        raise FileExistsError("Use a new directory to preserve a completed diagnostic")
    cfg = json.loads((pilot/"config.json").read_text())
    cfg["training"]["device"] = device
    seed = cfg["training"]["seed"]
    seed_everything(seed,cfg["training"]["threads"])
    x_cpu, metadata = select_training_windows(pilot)
    n = x_cpu.shape[1]//cfg["model"]["patch_size"]
    masks_cpu = span_mask(4,n,0.25,1,torch.Generator().manual_seed(cfg["evaluation_seed"]))
    x, masks = x_cpu.to(device), masks_cpu.to(device)
    donors = x.roll(-1,0)
    save_json(output/"config.json", {"pilot_config":cfg,"steps":steps,"training_windows":metadata,
              "fixed_mask_seed":cfg["evaluation_seed"],"training_only":True,
              "normalization": "unchanged original pilot training statistics",
              "cases": [1,4],"architecture_changes":False})
    save_json(output/"environment.json", {"torch":torch.__version__,"numpy":np.__version__,
              "device":torch.cuda.get_device_name(0) if device=="cuda" else "cpu",
              "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}})
    np.savez_compressed(output/"inputs.npz",x=x_cpu.numpy(),mask=masks_cpu.numpy())
    summary, curves = {}, []
    log_steps = {0,1,10,25,50,100,200,400,800,1200,1600,steps}
    for count in (1,4):
        source, mask, donor = x[:count], masks[:count], donors[:count]
        seed_everything(seed,cfg["training"]["threads"])
        model = make_eeg_model(cfg,"masked_short",x_cpu.shape[1:]).to(device)
        initial_checks = input_checks(model,source,mask,donor,seed)
        initial_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["training"]["lr"],
                                     weight_decay=cfg["training"]["weight_decay"])
        start = time.perf_counter()
        for step in range(steps+1):
            if step in log_steps:
                model.eval()
                with torch.no_grad(): loss = float(model(source,mask)["loss"])
                curves.append({"windows":count,"updates":step,"fixed_mask_mse":loss})
                print(f"{count} fixed window(s), update {step}: MSE={loss:.6f}",flush=True)
            if step == steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result = model(source,mask)
            if not torch.isfinite(result["loss"]):
                raise RuntimeError("Nonfinite diagnostic loss")
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["grad_clip"])
            optimizer.step()
        elapsed = time.perf_counter()-start
        checks = input_checks(model,source,mask,donor,seed)
        gradients = gradient_checks(model,source,mask)
        new_cpu = span_mask(count,n,0.25,1,torch.Generator().manual_seed(cfg["evaluation_seed"]+1))
        with torch.no_grad():
            final = model(source,mask)
            prediction = final["prediction"].cpu().numpy()
            fresh_mask_mse = float(model(source,new_cpu.to(device))["loss"])
        torch.save(model.state_dict(),output/f"fixed_{count}_weights.pt")
        target = patchify(x_cpu[:count],model.patch_size).numpy()[masks_cpu[:count].numpy()]
        singular = np.linalg.svd(target.astype(np.float64)-target.mean(0,dtype=np.float64),compute_uv=False)
        floor = float(np.sum(singular[cfg["model"]["d_model"]:]**2)/target.size)
        pilot_model = make_eeg_model(cfg,"masked_short",x_cpu.shape[1:]).to(device)
        pilot_model.load_state_dict(torch.load(pilot/"masked_short/best.pt",map_location=device,weights_only=True))
        pilot_checks = input_checks(pilot_model,source,mask,donor,seed)
        pilot_gradients = gradient_checks(pilot_model,source,mask)
        assert checks["masked_value_perturbation_max_change"] == 0
        assert gradients["masked_input_gradient_l2"] == 0
        assert any(not torch.equal(initial_state[k],model.state_dict()[k].cpu()) for k in initial_state)
        summary[str(count)] = {"baselines":baseline_scores(x_cpu[:count],masks_cpu[:count],model.patch_size),
              "initial":initial_checks,"fixed_fit":checks,"gradients":gradients,
              "original_pilot":pilot_checks,"original_pilot_gradients":pilot_gradients,
              "new_mask_same_training_windows_mse":fresh_mask_mse,
              "new_mask_baselines":baseline_scores(x_cpu[:count],new_cpu,model.patch_size),
              "affine_linear_head_rank_lower_bound_mse":floor,
              "parameters":sum(p.numel() for p in model.parameters()),"updates":steps,"training_seconds":elapsed}
        np.savez_compressed(output/f"fixed_{count}_predictions.npz",prediction=prediction,new_mask=new_cpu.numpy())
        save_json(output/"summary.json",summary)
        pd.DataFrame(curves).to_csv(output/"learning_curves.csv",index=False)
    plot_checks(output,summary,curves,x_cpu,masks_cpu,cfg)
    write_report(output,summary,steps,metadata)
    return summary


def run_resampled_control(pilot, output, steps=2000, device="cuda"):
    """Keep the four EEG windows fixed, but redraw short masks for every update."""
    pilot,output=Path(pilot),Path(output)
    if (output/"resampled_control.json").exists():
        raise FileExistsError("Resampled control already exists; use a new output directory")
    cfg=json.loads((pilot/"config.json").read_text())
    metadata=json.loads((output/"config.json").read_text())["training_windows"]
    train_ids={p["participant_id"] for p in json.loads((pilot/"participants.json").read_text())["train"]}
    assert all(row["participant_id"] in train_ids for row in metadata)
    with np.load(output/"inputs.npz") as data:
        x_cpu=torch.from_numpy(data["x"])
        fixed_cpu=torch.from_numpy(data["mask"])
    seed=cfg["training"]["seed"]
    seed_everything(seed,cfg["training"]["threads"])
    model=make_eeg_model(cfg,"masked_short",x_cpu.shape[1:]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["lr"],weight_decay=cfg["training"]["weight_decay"])
    x,fixed=x_cpu.to(device),fixed_cpu.to(device)
    generator=torch.Generator().manual_seed(seed+2)
    n=x.shape[1]//model.patch_size
    fresh_cpu=span_mask(4,n,.25,1,torch.Generator().manual_seed(cfg["evaluation_seed"]+1))
    curves=[]
    start=time.perf_counter()
    for step in range(steps+1):
        if step in {0,1,10,25,50,100,200,400,800,1200,1600,steps}:
            model.eval()
            with torch.no_grad():
                fixed_loss=float(model(x,fixed)["loss"])
                fresh_loss=float(model(x,fresh_cpu.to(device))["loss"])
            curves.append({"updates":step,"original_fixed_mask_mse":fixed_loss,"new_mask_mse":fresh_loss})
            print(f"4 windows, resampled masks, update {step}: fixed={fixed_loss:.6f}, new={fresh_loss:.6f}",flush=True)
        if step==steps:break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result=model(x,generator=generator)
        if not torch.isfinite(result["loss"]):raise RuntimeError("Nonfinite resampled loss")
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["grad_clip"])
        optimizer.step()
    elapsed=time.perf_counter()-start
    model.eval()
    evaluation=[]
    for mask_seed in range(cfg["evaluation_seed"]+1,cfg["evaluation_seed"]+9):
        mask_cpu=span_mask(4,n,.25,1,torch.Generator().manual_seed(mask_seed))
        with torch.no_grad():mse=float(model(x,mask_cpu.to(device))["loss"])
        evaluation.append({"mask_seed":mask_seed,"neural_mse":mse,**baseline_scores(x_cpu,mask_cpu,model.patch_size)})
    result={"updates":steps,"training_seconds":elapsed,"training_mask_seed":seed+2,
            "fixed_probe":input_checks(model,x,fixed,x.roll(-1,0),seed),
            "new_mask_mse":curves[-1]["new_mask_mse"],"gradients":gradient_checks(model,x,fixed),
            "independent_evaluation_masks":evaluation,
            "eight_mask_mean_mse":{key:float(np.mean([r[key] for r in evaluation]))
                                   for key in ("neural_mse","training_mean","linear","nearest")}}
    save_json(output/"resampled_control.json",result)
    save_json(output/"resampled_environment.json",{"torch":torch.__version__,"steps":steps,
              "diagnostic_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "note":"Only original training windows; no validation/test EEG or classifier evaluation."})
    pd.DataFrame(curves).to_csv(output/"resampled_learning_curves.csv",index=False)
    torch.save(model.state_dict(),output/"resampled_4_weights.pt")
    fig,ax=plt.subplots(figsize=(7,4))
    for key in ("original_fixed_mask_mse","new_mask_mse"):
        ax.plot([r["updates"]+1 for r in curves],[r[key] for r in curves],marker="o",label=key)
    for key in ("training_mean","linear"):
        ax.axhline(result["eight_mask_mean_mse"][key],ls=":",color="gray" if key=="training_mean" else "green",
                   label=f"{key}, 8-mask average")
    ax.set(xscale="log",yscale="log",xlabel="Updates + 1 (log scale)",ylabel="MSE (log scale)",
           title="Four training windows; a new mask on every update")
    ax.legend(fontsize=8)
    finish(fig,output/"resampled_learning_curves.png")
    run_unfitted_check(pilot,output,device)
    return result


def run_unfitted_check(pilot,output,device="cuda"):
    """Inference on a second cached window per fitted training participant; no cohort test data."""
    pilot,output=Path(pilot),Path(output)
    cfg=json.loads((pilot/"config.json").read_text())
    metadata=json.loads((output/"config.json").read_text())["training_windows"]
    train_ids={p["participant_id"] for p in json.loads((pilot/"participants.json").read_text())["train"]}
    assert all(row["participant_id"] in train_ids for row in metadata)
    with np.load(pilot/"normalization.npz") as stats:mean,std=stats["mean"],stats["std"]
    windows,selected=[],[]
    for row in metadata:
        with np.load(row["cache"]) as data:
            windows.append((data["x"][1]-mean)/std)
            selected.append({"participant_id":row["participant_id"],"cached_epoch_index":1,
                             "source_start_sample":int(data["source_start"][1])})
    x_cpu=torch.from_numpy(np.stack(windows));x=x_cpu.to(device)
    model=make_eeg_model(cfg,"masked_short",x_cpu.shape[1:]).to(device).eval()
    model.load_state_dict(torch.load(output/"resampled_4_weights.pt",map_location=device,weights_only=True))
    n=x.shape[1]//model.patch_size
    scores=[]
    for seed in range(cfg["evaluation_seed"]+1,cfg["evaluation_seed"]+9):
        mask=span_mask(4,n,.25,1,torch.Generator().manual_seed(seed))
        with torch.no_grad():mse=float(model(x,mask.to(device))["loss"])
        scores.append({"mask_seed":seed,"neural_mse":mse,**baseline_scores(x_cpu,mask,model.patch_size)})
    result={"windows":selected,"training_only":True,"updates":0,"eight_masks":scores,
            "mean_mse":{key:float(np.mean([r[key] for r in scores]))
                        for key in ("neural_mse","training_mean","linear","nearest")},
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(output/"unfitted_window_check.json",result)
    summary=json.loads((output/"summary.json").read_text())
    steps=json.loads((output/"resampled_control.json").read_text())["updates"]
    write_report(output,summary,steps,metadata)
    return result


def plot_checks(output,summary,curves,x,mask,cfg):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,count in zip(axes,(1,4)):
        rows=[r for r in curves if r["windows"]==count]
        ax.plot([r["updates"]+1 for r in rows],[r["fixed_mask_mse"] for r in rows],marker="o",label="Neural fixed-window fit")
        for color,(name,value) in zip(("gray","orange","green"),summary[str(count)]["baselines"].items()):
            ax.axhline(value,ls=":",color=color,label=name)
        ax.axhline(summary[str(count)]["original_pilot"]["original"]["mse"],ls="--",color="purple",label="Original pilot checkpoint")
        ax.set(xscale="log",yscale="log",xlabel="Optimizer updates + 1 (log scale)",ylabel="Fixed training-mask MSE (log scale)",title=f"{count} fixed training window(s)")
        ax.legend(fontsize=7)
    finish(fig,output/"learning_curves.png")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=["original","zero_visible","shuffle_visible","other_window_visible"]
    for ax,count in zip(axes,(1,4)):
        values=[summary[str(count)]["fixed_fit"][name]["mse"] for name in labels]
        ax.bar(range(4),values)
        ax.set_xticks(range(4),labels,rotation=25,ha="right",fontsize=8)
        ax.set(ylabel="MSE against unchanged fixed targets",title=f"{count} window(s): visible-input ablations")
    finish(fig,output/"input_sensitivity.png")
    fig,axes=plt.subplots(2,1,figsize=(12,6))
    p=cfg["model"]["patch_size"];fs=cfg["sampling_hz"]
    times=np.arange(x.shape[1])/fs
    for ax,count in zip(axes,(1,4)):
        with np.load(output/f"fixed_{count}_predictions.npz") as data:
            predicted=data["prediction"][0].reshape(-1,p,x.shape[-1])[...,0].ravel()
        hidden=np.repeat(mask[0].numpy(),p)
        predicted[~hidden]=np.nan
        interp=fill_masked(x[0].numpy().reshape(-1,p,x.shape[-1]),mask[0].numpy())[...,0].ravel()
        interp[~hidden]=np.nan
        ax.plot(times,x[0,:,0],color="black",lw=.7,label="Target EEG")
        ax.plot(times,predicted,lw=1,label="Fixed-fit neural prediction")
        ax.plot(times,interp,lw=.8,label="Interpolation")
        ax.set(title=f"First training window; model fitted to {count} window(s)",xlabel="Seconds",ylabel="Normalized Fp1")
        ax.legend(fontsize=8)
    finish(fig,output/"reconstruction_examples.png")


def write_report(output,summary,steps,metadata):
    lines=["# EEG learning sanity checks","",f"Training-only diagnostic: {steps} updates on one and four fixed 8-second EEG windows, using fixed 25% short-patch masks. Same 56,411-parameter model, initialization, AdamW settings and original training normalization as the pilot. No held-out EEG is loaded and no model settings are tuned. Participants: {', '.join(r['participant_id'] for r in metadata)}.","",
           "These are memorization/input-sensitivity tests, not validation, classification or generalization results. Mask/optimizer seeds, exact selected windows and normalized inputs are saved.","",
           "| Fixed windows | Initial MSE | Pilot checkpoint MSE | Fixed-fit MSE | Training mean | Interpolation | New-mask MSE (same windows) |","| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for count,result in summary.items():
        lines.append(f"| {count} | {result['initial']['original']['mse']:.6f} | {result['original_pilot']['original']['mse']:.6f} | {result['fixed_fit']['original']['mse']:.6f} | {result['baselines']['training_mean']:.6f} | {result['baselines']['linear']:.6f} | {result['new_mask_same_training_windows_mse']:.6f} |")
    lines += ["","## Visible-input dependence","","All ablations change visible patches only. Hidden targets stay unchanged. Shuffle permutes visible patches within each window; other-window replacement uses another training participant, never validation/test data.","",
              "| Windows | Original MSE | Zero visible MSE | Shuffled visible MSE | Other-window visible MSE | Masked-value perturbation max prediction change |","| --- | ---: | ---: | ---: | ---: | ---: |"]
    for count,result in summary.items():
        r=result['fixed_fit']
        lines.append(f"| {count} | {r['original']['mse']:.6f} | {r['zero_visible']['mse']:.6f} | {r['shuffle_visible']['mse']:.6f} | {r['other_window_visible']['mse']:.6f} | {r['masked_value_perturbation_max_change']:.1f} |")
    lines += ["","Prediction-change RMS values and visible/hidden input gradients are in `summary.json`. Hidden-value perturbation must leave predictions unchanged, and the gradient of predictions with respect to hidden inputs must be zero. Finite input sensitivity does not prove the model learned interpolation.","",
              "The optional affine-head rank bound is the minimum MSE obtainable by any rank-d_model centered output matrix on these targets. It follows from the shared linear output head and is a necessary capacity bound, not a guarantee that the encoder can attain it.","",
              "A low fixed-mask training loss can reflect memorizing positions/windows. New masks on the same fitted windows are a harder diagnostic but still do not test held-out participant generalization. Data-scaling conclusions require more evidence than these checks.","",
              f"![Learning curves]({output.resolve()/'learning_curves.png'})",f"![Input dependence]({output.resolve()/'input_sensitivity.png'})",f"![Examples]({output.resolve()/'reconstruction_examples.png'})","",
              "```bash","CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \\",f"  .venv-gpu/bin/python -m temporal_ssl.eeg_learning_checks --pilot results/eeg_pilot --output results/eeg_learning_checks --steps {steps}","```","Use a new output directory for another run."]
    if (output/"resampled_control.json").exists():
        result=json.loads((output/"resampled_control.json").read_text())
        mean=result["eight_mask_mean_mse"]
        lines += ["","## Four-window resampled-mask control","",
                  f"Same four windows, initialization, architecture, optimizer and {result['updates']} updates; only the mask changes each update, as in the original pilot. No hyperparameter search.","",
                  f"Original fixed-probe MSE: **{result['fixed_probe']['original']['mse']:.6f}**. New-mask MSE on the same windows: **{result['new_mask_mse']:.6f}**.","",
                  f"Across eight independently sampled evaluation masks on those same four training windows: neural MSE **{mean['neural_mse']:.6f}**, interpolation **{mean['linear']:.6f}**, nearest **{mean['nearest']:.6f}**, training mean **{mean['training_mean']:.6f}**. The eight masks assess mask variability, not independent participant generalization.","",
                  f"![Resampled-mask learning]({output.resolve()/'resampled_learning_curves.png'})"]
    if (output/"unfitted_window_check.json").exists():
        result=json.loads((output/"unfitted_window_check.json").read_text())["mean_mse"]
        lines += ["","## Unfitted windows from the same training participants","",
                  f"Inference only on cached window 1 (rather than fitted window 0) from each of the same four training participants, using eight evaluation masks. No additional optimization or held-out participant EEG. Neural MSE **{result['neural_mse']:.6f}**, interpolation **{result['linear']:.6f}**, nearest **{result['nearest']:.6f}**, training mean **{result['training_mean']:.6f}**.","",
                  "This distinguishes fitting the four waveforms from transfer to another window. It is still not an independent-subject validation result."]
    lines += ["","## Interpretation","",
              "Low loss on fixed windows/masks verifies that the optimizer, gradient path and model can fit these EEG targets; it does not verify learned interpolation. The resampled-mask control asks whether fitting survives changes of hidden positions on the same waveforms. The unfitted-window check asks whether that behavior transfers to different waveforms from the same participants. Neither is a held-out-subject benchmark.","",
              f"Each tiny-case window is presented {steps} times, once per update. The original pilot budget presented each window only 20 times (its selected checkpoint can be earlier). Thus this deliberately changes repetition/exposure as well as dataset size; it cannot establish that adding data hurts, or that a longer full-data run would necessarily solve the problem.","",
              "These checks do not establish a need for foundation-model-scale data. If new-mask fitted-window errors are low but unfitted-window errors are high, focus next on learning a transferable reconstruction rule, with controlled training exposure and patch/context design. A temporally labeled EEG dataset can improve the scientific evaluation target, but switching datasets does not itself fix this learning issue."]
    report="\n".join(lines)+"\n"
    (output/"report.md").write_text(report.replace(str(output.resolve())+"/",""))
    destination=Path(__file__).resolve().parents[1]/"reports/eeg_learning_checks.md"
    assets=destination.parent/"eeg_learning_checks_assets"
    assets.mkdir(exist_ok=True)
    for name in ("learning_curves.png","input_sensitivity.png","reconstruction_examples.png","resampled_learning_curves.png"):
        if (output/name).exists():shutil.copyfile(output/name,assets/name)
    destination.write_text(report.replace(str(output.resolve())+"/","eeg_learning_checks_assets/"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",default="results/eeg_pilot")
    parser.add_argument("--output",default="results/eeg_learning_checks")
    parser.add_argument("--steps",type=int,default=2000)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--resampled-only",action="store_true",help="Add the resampled control after completed fixed-mask diagnostics")
    args=parser.parse_args()
    if args.steps<1:parser.error("--steps must be positive")
    if not args.resampled_only:
        run_checks(args.pilot,args.output,args.steps,args.device)
    run_resampled_control(args.pilot,args.output,args.steps,args.device)


if __name__=="__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    main()
