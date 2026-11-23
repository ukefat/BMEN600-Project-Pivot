"""One-/five-patch latent targets, using independent shared-encoder windows."""
import copy,json
from pathlib import Path
import pandas as pd
from .plots import plt,finish
from . import eeg_latent_experiment as experiment
from .experiment import save_json


def report():
    rows=[]
    for context,root in [('global',Path('results/eeg_latent')),(1,Path('results/eeg_latent_local1')),(5,Path('results/eeg_latent_local5'))]:
        if context!='global':experiment.diagnose_controls(root,json.loads((root/'protocol.json').read_text()))
        downstream=json.loads((root/'control_diagnostics.json').read_text())
        h=json.loads((root/'history.json').read_text());selected=json.loads((root/'selected.json').read_text())
        diagnostic=next(r for r in h if r['step']==selected['step'])
        for name in ['latent','latent_final']:
            metric_row=diagnostic if name=='latent' else h[-1]
            scores=json.loads((root/name/'classification.json').read_text())['test']
            rows.append({'target_context':str(context),'checkpoint':name,'balanced_accuracy':scores['balanced_accuracy'],
                'macro_f1':scores['macro_f1'],'weights_step':selected['step'] if name=='latent' else h[-1]['step'],
                'target_patch_rank':metric_row['validation_patch_effective_rank'],
                'target_pooled_rank':metric_row['validation_pooled_effective_rank'],
                'target_temporal_variance':metric_row.get('validation_within_window_variance',downstream[name]['within_window_variance']),
                'masked_patch_rank':metric_row['validation_masked_patch_effective_rank'],
                'masked_pooled_rank':metric_row['validation_masked_pooled_effective_rank'],
                'downstream_full_window_rank':downstream[name]['pooled_effective_rank'],
                'downstream_full_window_temporal_variance':downstream[name]['within_window_variance']})
    f=pd.DataFrame(rows);f.to_csv('reports/eeg_latent_local.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for label,path in [('Global',Path('results/eeg_latent')),('One patch',Path('results/eeg_latent_local1')),('Five patches',Path('results/eeg_latent_local5'))]:
        h=pd.DataFrame(json.loads((path/'history.json').read_text()))
        for ax,key,title in zip(axes.flat,['validation_loss','validation_patch_effective_rank','validation_pooled_effective_rank','validation_masked_pooled_effective_rank'],['Combined validation objective','Local target patch rank','Local target pooled rank','Masked full-context pooled rank']):
            ax.plot(h.step,h[key],label=label);ax.set(title=title,xlabel='Updates');ax.legend()
    assets=Path('reports/eeg_latent_local_assets');assets.mkdir(exist_ok=True)
    finish(fig,assets/'diagnostics.png')
    lines=['# Local target neighborhoods for masked latent prediction','','Two follow-ups change only the unmasked target branch: one-patch (50 ms) or five-patch (250 ms) independent windows. The masked context branch and predictor retain full-window visible context. Shared parameters, no EMA/stop-gradient, same SIGReg applied across samples separately at each target position, same 25% masking / 50 ms patches, seed 600, full 4,894 training windows and 10,000 updates × 64 examples. No extra variance/sample objective. Original global-target results are preserved.','','Each target window is encoded independently through all Transformer layers and only its center output retained. Original absolute positional encodings are used. At sequence boundaries, out-of-range tokens are masked as keys/values; they are never replicated observations. Thus a five-patch target at an interior position reads exactly two preceding and two following patches, with no multilayer propagation beyond that boundary. Target outputs do not enter the predictor.','','Frozen classifiers use the same full-context unmasked encoder, first-four-second input and original participant averaging/scaling. Target branch locality does not change feature extraction. All models are validation-selected by combined objective. Diagnostics in the table describe target embeddings; they are not automatically the same as full-context downstream embeddings. This is a one-seed diagnostic on the reused 18-person test set.','','| Target context | Checkpoint | Test balanced accuracy | Macro-F1 | Weights update | Target patch rank | Target pooled rank |','|---|---|---:|---:|---:|---:|---:|']
    for r in f.itertuples():lines.append(f'| {r.target_context} | {r.checkpoint} | {r.balanced_accuracy:.1%} | {r.macro_f1:.3f} | {r.weights_step} | {r.target_patch_rank:.2f} | {r.target_pooled_rank:.2f} |')
    lines+=['','Target diagnostics, objective components and masked-context variance/rank are saved every 500 updates in each `results/eeg_latent_local{1,5}/history.json`. Initial, selected and final weights, predictions and classifiers remain alongside them. The untrained reference is 35.7% and paired waveform reference 34.9%. CSV: [full comparison](eeg_latent_local.csv).','','Reproduce: `CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv-gpu/bin/python -m temporal_ssl.eeg_latent_local`. Completed runs resume without extra training.']
    lines+=['','Local targets restore temporal variation: at selected checkpoints, target within-window variance is approximately 0.00266 (global), 0.4825 (one patch), and 0.5530 (five patches). Full-context feature temporal variance also increases, but pooled effective rank stays low. Diagnosis point estimates do not show a convincing gain over the untrained or waveform reference. This is evidence that fixing the measured broadcast behavior is not sufficient for diagnostic utility under the current readout; it does not prove that changing pooling would solve the problem.','', 'Masking remains 25% with a requested 50 ms masked span. The target neighborhood is a separate parameter; it does not change patch size or masked-span length.','','![Target and context diagnostics](eeg_latent_local_assets/diagnostics.png)','', 'The objective curves use different learned targets and should not be ranked as reconstruction accuracy. Final-checkpoint rows show final weights and corresponding diagnostics; primary rows follow validation selection.']
    Path('reports/eeg_latent_local.md').write_text('\n'.join(lines)+'\n');print(f.to_string(index=False),flush=True)


def main():
    original=json.loads(Path('configs/eeg_latent.json').read_text())
    writer=experiment.report;experiment.report=lambda *_:None
    try:
        for context in [1,5]:
            cfg=copy.deepcopy(original);cfg['target_context']=context;cfg['output']=f'results/eeg_latent_local{context}'
            path=Path(f'configs/eeg_latent_local{context}.json');save_json(path,cfg)
            experiment.run(path)
    finally:experiment.report=writer
    report()


if __name__=='__main__':main()
