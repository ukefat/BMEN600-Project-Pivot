"""Frozen AD/HC readouts for the centralized waveform study; no SSL retraining."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,f1_score,recall_score,roc_auc_score,confusion_matrix
from .experiment import save_json
from .plots import plt,finish


def run():
    source=Path('results/eeg_reconstruction_study');root=Path('results/eeg_binary');root.mkdir(exist_ok=True)
    entries=[]
    provenance=json.loads((source/'checkpoint_provenance.json').read_text())
    for r in provenance:
        p=Path(r['checkpoint']).parent/'classification/participant_features.npz'
        entries.append((r['model'],r['seed'],p))
    for seed in [600,601,602]:
        entries.append(('untrained_causal',seed,source/'context80'/f'seed_{seed}'/'untrained_classification/participant_features.npz'))
        entries.append(('untrained_masked',seed,Path('results/eeg_factorial/untrained')/f'seed_{seed}'/'classification/participant_features.npz'))
    entries.append(('handcrafted_spectral',None,Path('results/eeg_pilot/spectral/participant_features.npz')))
    rows=[];all_predictions=[];counts={};seen={}
    for name,seed,path in entries:
        data={}
        with np.load(path) as a:
            for part in ['train','validation','test']:
                keep=np.isin(a[part+'_y'],['AD','HC'])
                ids=a[part+'_ids'][keep];x=a[part+'_x'][keep];y=(a[part+'_y'][keep]=='AD').astype(int)
                data[part]=(ids,x,y)
                if part in seen:np.testing.assert_array_equal(ids,seen[part])
                else:seen[part]=ids
                counts[part]={'participants':len(y),'AD':int(y.sum()),'HC':int((y==0).sum())}
        assert not(set(data['train'][0])&set(data['test'][0]) or set(data['train'][0])&set(data['validation'][0]) or set(data['validation'][0])&set(data['test'][0]))
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='lbfgs',max_iter=5000,tol=1e-6,random_state=600))
        model.fit(data['train'][1],data['train'][2])
        np.testing.assert_allclose(model[0].mean_,data['train'][1].mean(0,dtype=np.float64),atol=1e-10)
        output=root/name/(f'seed_{seed}' if seed is not None else 'fixed');output.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,output/'classifier.joblib')
        predictions=[]
        for part in ['validation','test']:
            ids,x,y=data[part];pred=model.predict(x);prob=model.predict_proba(x)[:,1]
            recall=recall_score(y,pred,labels=[0,1],average=None,zero_division=0)
            rows.append({'model':name,'seed':seed,'split':part,'balanced_accuracy':balanced_accuracy_score(y,pred),
                'macro_f1':f1_score(y,pred,average='macro',labels=[0,1],zero_division=0),'roc_auc':roc_auc_score(y,prob),
                'recall_HC':recall[0],'recall_AD':recall[1]})
            for pid,truth,guess,p in zip(ids,y,pred,prob):predictions.append({'model':name,'seed':seed,'split':part,'participant_id':pid,'truth':int(truth),'prediction':int(guess),'probability_AD':p})
            pd.DataFrame(confusion_matrix(y,pred,labels=[0,1]),index=['HC','AD'],columns=['HC','AD']).to_csv(output/f'{part}_confusion.csv')
        pd.DataFrame(predictions).to_csv(output/'predictions.csv',index=False);all_predictions.extend(predictions)
    frame=pd.DataFrame(rows);frame.to_csv(root/'scores.csv',index=False);pd.DataFrame(all_predictions).to_csv(root/'predictions.csv',index=False)
    summary=frame[frame.split=='test'].groupby('model').agg(balanced_accuracy=('balanced_accuracy','mean'),ba_seed_sd=('balanced_accuracy','std'),macro_f1=('macro_f1','mean'),f1_seed_sd=('macro_f1','std'),roc_auc=('roc_auc','mean'),auc_seed_sd=('roc_auc','std')).reset_index()
    summary.to_csv(root/'summary.csv',index=False)
    previous=pd.read_csv(source/'classification_summary.csv')[['model','balanced_accuracy']].rename(columns={'balanced_accuracy':'three_class_ba'})
    comparison=summary.merge(previous,on='model');comparison.to_csv(root/'three_vs_two.csv',index=False)
    save_json(root/'protocol.json',{'counts':counts,'positive_class':'AD','negative_class':'HC','FTD_in_classifier':False,'FTD_in_original_SSL':True,'ssl_retrained':False,'C':1,'labels_used_in_checkpoint_selection':False,'source_features':[str(e[2]) for e in entries]})
    fig,ax=plt.subplots(figsize=(12,5));ax.bar(summary.model,summary.balanced_accuracy,yerr=summary.ba_seed_sd.fillna(0),capsize=3)
    ax.axhline(.5,color='gray',ls=':',label='Chance');ax.set(ylim=(0,1),ylabel='AD/HC test balanced accuracy');ax.tick_params(axis='x',labelrotation=40);ax.legend();finish(fig,root/'classification.png')
    def table(f):
        lines=['| '+' | '.join(f.columns)+' |','|'+'|'.join(['---']*len(f.columns))+'|']
        for row in f.itertuples(index=False,name=None):lines.append('| '+' | '.join('—' if pd.isna(v) else f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(lines)
    text=f'''# AD versus healthy-control frozen EEG classification

Binary diagnosis readouts for every configuration in the centralized waveform study. Existing frozen encoder features are reused; no SSL retraining or new checkpoint selection. FTD is excluded from classifier fitting, validation and test scoring. **FTD training EEG remains part of the original label-free SSL pretraining.** This tests whether the same representations support AD/HC discrimination, not AD/HC-only pretraining.

Counts: {counts}. The same participant-disjoint split is preserved, rather than resplitting to improve scores. Positive class AD, negative class HC. Each participant still has one mean-pooled feature vector from the same 24 first-four-second windows. StandardScaler and fixed C=1 logistic regression are refit on AD/HC training participants only. No classifier hyperparameter sweep, threshold tuning, or test-based choices. Test BA, macro-F1 and ROC-AUC are reported. Chance balanced accuracy is now **50%**, compared with 33.3% for the earlier three-class task, so absolute accuracies cannot be interpreted as a direct model improvement across tasks.

## Test results

{table(summary)}

Mean and sample SD over the three paired encoders; spectral features have one deterministic classifier and no seed SD. The test cohort is smaller after excluding FTD, and has already been inspected in earlier experiments. These remain exploratory fixed-split results, not independent confirmation or a clinical diagnostic evaluation.

![Binary classification](eeg_binary_assets/classification.png)

[Per-seed scores](eeg_binary_assets/scores.csv), [summary](eeg_binary_assets/summary.csv), [three-class versus binary point estimates](eeg_binary_assets/three_vs_two.csv). Per-participant probabilities, classifiers and confusion matrices are in `{root}`. Reconstruction scores do not change when only downstream diagnosis labels change.

Reproduce without SSL retraining: `.venv-gpu/bin/python -m temporal_ssl.eeg_binary`.
'''
    import shutil
    assets=Path('reports/eeg_binary_assets');assets.mkdir(exist_ok=True)
    for file in ['scores.csv','summary.csv','three_vs_two.csv','classification.png','protocol.json']:shutil.copyfile(root/file,assets/file)
    Path('reports/eeg_binary.md').write_text(text);(root/'report.md').write_text(text.replace('eeg_binary_assets/',''))
    print(counts);print(summary.to_string(index=False))


if __name__=='__main__':run()
