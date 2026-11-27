"""CPU-only reproduction from committed numerical evidence. Never trains or overwrites.

Requires NumPy, pandas, Matplotlib; optional --overview needs Inkscape.
Use a new --output directory for every invocation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def module(path):
    spec=importlib.util.spec_from_file_location('report_builder',path)
    imported=importlib.util.module_from_spec(spec);spec.loader.exec_module(imported)
    return imported


def run(output, overview=False):
    started=time.perf_counter()
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    used={}
    def read(relative):
        path=ROOT/relative;used[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)
    def save(fig,name):
        fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=180);fig.savefig(output/(name+'.pdf'));plt.close(fig)
    central='reports/eeg_reconstruction_extended_assets/'
    classes=read(central+'classification.csv').query("split == 'test'")
    summary=classes.groupby('model').agg(balanced_accuracy=('balanced_accuracy','mean'),
        ba_seed_sd=('balanced_accuracy','std'),macro_f1=('macro_f1','mean')).reset_index()
    expected=read(central+'classification_summary.csv').sort_values('model').reset_index(drop=True)
    for key in ['balanced_accuracy','ba_seed_sd','macro_f1']:
        np.testing.assert_allclose(summary[key],expected[key],atol=1e-12,rtol=0,equal_nan=True)
    summary.to_csv(output/'diagnostic_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4.6));x=np.arange(len(summary))
    ax.bar(x,summary.balanced_accuracy*100,yerr=summary.ba_seed_sd.fillna(0)*100,capsize=2)
    ax.axhline(100/3,color='gray',ls=':');ax.set(xticks=x,xticklabels=summary.model,
        ylabel='Test balanced accuracy (%)',ylim=(0,100));ax.tick_params(axis='x',labelrotation=65)
    save(fig,'diagnostic_performance')
    recon=read(central+'reconstruction.csv').query("split == 'test'")
    r=recon.groupby('model').agg(neural_mse=('neural_mse','mean'),mse_seed_sd=('neural_mse','std'),
        mean_mse=('mean_mse','mean'),interpolation_mse=('interpolation_mse','mean'),
        persistence_mse=('persistence_mse','mean')).reset_index()
    expected=read(central+'reconstruction_summary.csv').sort_values('model').reset_index(drop=True)
    for key in r.columns[1:]:np.testing.assert_allclose(r[key],expected[key],atol=1e-12,rtol=0,equal_nan=True)
    r.to_csv(output/'waveform_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,prefix in zip(axes,['masked','causal']):
        q=r[r.model.str.startswith(prefix)];x=np.arange(len(q))
        ax.bar(x,q.neural_mse,yerr=q.mse_seed_sd,capsize=2,label='Neural')
        ax.plot(x,q.mean_mse,'o',label='Training-channel mean')
        ax.plot(x,q.interpolation_mse if prefix=='masked' else q.persistence_mse,'s',
                label='Interpolation' if prefix=='masked' else 'Persistence')
        ax.set(xticks=x,xticklabels=q.model,ylabel='Task-specific normalized test MSE',title=prefix)
        ax.tick_params(axis='x',labelrotation=65);ax.legend(fontsize=7)
    save(fig,'waveform_prediction')
    spectral=read('reports/eeg_window_spectral_assets/scores.csv')
    spectral.query("split == 'test'").to_csv(output/'spectral_scores.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True);bands=['delta','theta','alpha','beta','gamma']
    for ax,pooling in zip(axes,['mean','mean_std']):
        q=spectral[(spectral.split=='test')&(spectral.pooling==pooling)]
        for row in q.to_dict('records'):ax.plot(bands,[row[b] for b in bands],marker='o',label=row['model'])
        base=spectral[(spectral.model=='training_mean')&(spectral.split=='test')].iloc[0]
        ax.plot(bands,base[bands].to_numpy(float),'k--',label='Training mean');ax.axhline(0,color='gray',lw=.7)
        ax.set(title=pooling,ylabel='Window-level test R²');ax.legend(fontsize=7)
    save(fig,'spectral_accessibility')
    split=read('reports/submission_assets/participant_split.csv')
    qc=read('reports/eeg_supplement_assets/qc_by_participant.csv')
    assert set(split.participant_id)==set(qc.participant_id) and len(split)==88
    cohort=pd.crosstab(split.split,split.diagnosis).reindex(['train','validation','test'])
    cohort['participants']=cohort.sum(axis=1)
    for key in ['usable_windows','selected_panel_windows']:
        cohort[key]=qc.groupby('split')[key].sum()
    cohort.to_csv(output/'cohort.csv')
    qc_totals=read('reports/eeg_supplement_assets/qc_totals.csv')
    for row in qc_totals.itertuples():
        q=qc if row.split=='all' else qc[qc.split==row.split]
        for field in qc_totals.columns[1:]:
            np.testing.assert_allclose(q[field].sum(),getattr(row,field),rtol=0,atol=1e-9)
    qc_totals.to_csv(output/'qc_counts.csv',index=False)
    pred=read('reports/eeg_supplement_assets/participant_predictions.csv').query("split == 'test'")
    ids=split[split.split=='test'].sort_values('participant_id');truth=ids.diagnosis.to_numpy()
    names=sorted(pred.model.unique());cubes=[]
    for name in names:
        seeds=[]
        for seed in [600,601,602]:
            q=pred[(pred.model==name)&(pred.seed.isna() if name=='handcrafted_spectral' else pred.seed==seed)].sort_values('participant_id')
            np.testing.assert_array_equal(q.participant_id,ids.participant_id)
            np.testing.assert_array_equal(q.truth,truth)
            seeds.append((q.prediction.to_numpy()==truth).astype(float))
        cubes.append(seeds)
    cube=np.asarray(cubes);draws=np.zeros((len(names),3,5000));rng=np.random.default_rng(5600)
    labels=['AD','FTD','HC']
    for label in labels:
        ix=np.flatnonzero(truth==label);sample=rng.choice(ix,(5000,len(ix)),replace=True)
        draws+=cube[:,:,sample].mean(-1)/3
    comparisons=read('reports/eeg_supplement_assets/paired_bootstrap.csv')
    for row in comparisons.itertuples():
        difference=draws[names.index(row.model)]-draws[names.index(row.reference)]
        lo,hi=np.quantile(difference.mean(0),[.025,.975])
        np.testing.assert_allclose([lo,hi],[row.lower_95,row.upper_95],rtol=0,atol=1e-12)
        delta=np.mean([cube[names.index(row.model),:,truth==label].mean()-cube[names.index(row.reference),:,truth==label].mean() for label in labels])
        np.testing.assert_allclose(delta,row.difference,rtol=0,atol=1e-12)
    comparisons.to_csv(output/'paired_diagnostic_intervals.csv',index=False)
    ftd=pred[pred.truth=='FTD'].groupby(['model','seed'],dropna=False).prediction.value_counts().unstack(fill_value=0)
    ftd=ftd.reindex(columns=labels,fill_value=0).reset_index()
    ftd.to_csv(output/'ftd_counts.csv',index=False)
    old=read('reports/eeg_supplement_assets/ftd_counts_per_seed.csv')
    np.testing.assert_array_equal(ftd[labels],old[labels])
    matrices=read('reports/eeg_supplement_assets/confusion_matrices_per_seed.csv')
    for (name,seed),q in pred.groupby(['model','seed'],dropna=False):
        count=pd.crosstab(q.truth,q.prediction).reindex(index=labels,columns=labels,fill_value=0)
        stored=matrices[(matrices.model==name)&(matrices.seed.isna() if pd.isna(seed) else matrices.seed==seed)].set_index('true_diagnosis').loc[labels,labels]
        np.testing.assert_array_equal(count,stored)
    matrices.to_csv(output/'confusion_matrices.csv',index=False)
    # Original synthetic figure builder validates every aggregate against seed rows.
    synthetic=module(ROOT/'reports/synthetic_summary_assets/build_figures.py')
    synthetic.ROOT=output;synthetic.run()
    if overview:
        overview_module=module(ROOT/'reports/study_overview_assets/build_figure.py')
        overview_module.SVG=output/'study_overview.svg';overview_module.PNG=output/'study_overview.png'
        overview_module.build()
    provenance=dict(source_sha256=used,seconds=time.perf_counter()-started,
        data_access='committed numerical evidence only; no caches, checkpoints, fitted readouts or training',
        verified=['diagnostic means/seed SDs','waveform means/seed SDs','13 paired bootstrap intervals',
                  'FTD counts and per-seed confusion matrices','QC totals','342 synthetic aggregate rows'])
    (output/'verification.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'CPU reproduction completed in {provenance["seconds"]:.2f} s: {output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/submission_preview')
    parser.add_argument('--overview',action='store_true',help='Also regenerate editable overview; requires Inkscape')
    args=parser.parse_args();run(args.output,args.overview)
