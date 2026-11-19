"""Two matched latent-span follow-ups, with separate artifacts and reports."""
import copy,json,shutil,re
from pathlib import Path
import pandas as pd
from . import eeg_latent_experiment as experiment
from .experiment import save_json


def main():
    original=json.loads(Path('configs/eeg_latent.json').read_text())
    writer=experiment.report
    # The general runner's report destination belongs to the original 50 ms run.
    for milliseconds in [100,200]:
        cfg=copy.deepcopy(original);cfg['mask_span']=milliseconds//50;cfg['output']=f'results/eeg_latent_span{milliseconds}'
        path=Path(f'configs/eeg_latent_span{milliseconds}.json');save_json(path,cfg)
        experiment.report=lambda *_:None
        try:experiment.run(path)
        finally:experiment.report=writer
        # Render with span-specific labels while preserving the original report/assets.
        reports=Path('reports');backup=Path(cfg['output'])/'report_backup';backup.mkdir(exist_ok=True)
        shutil.copyfile(reports/'eeg_latent.md',backup/'eeg_latent.md')
        shutil.copytree(reports/'eeg_latent_assets',backup/'eeg_latent_assets',dirs_exist_ok=True)
        try:
            writer(Path(cfg['output']),cfg)
            assets=f'eeg_latent_span{milliseconds}_assets'
            shutil.copytree(reports/'eeg_latent_assets',reports/assets,dirs_exist_ok=True)
            text=(reports/'eeg_latent.md').read_text().replace('eeg_latent_assets/',assets+'/')
            text=text.replace('span one patch (adjacent masks can merge)',f'requested span {milliseconds} ms ({milliseconds//50} patches; adjacent masks can merge)')
            text=re.sub(r'This run did \*\*not\*\*.*?No extra regularizer or objective was added in response to these results.', 'This is a matched-span follow-up. Interpret the measured classification and collapse statistics below; no extra regularizer or objective was added.',text,flags=re.S)
            text=text.replace('## Initial results','The waveform row is the original 50 ms reference, not a newly trained matched-span waveform model. This follow-up changes only the latent model’s mask span; it retains the original SIGReg and predictor settings.\n\n## Initial results')
            text=text.replace('--config configs/eeg_latent.json',f'--config configs/eeg_latent_span{milliseconds}.json')
            (reports/f'eeg_latent_span{milliseconds}.md').write_text(text)
            (Path(cfg['output'])/'report.md').write_text(text.replace(assets+'/',''))
        finally:
            shutil.copyfile(backup/'eeg_latent.md',reports/'eeg_latent.md')
            shutil.copytree(backup/'eeg_latent_assets',reports/'eeg_latent_assets',dirs_exist_ok=True)
    rows=[]
    for span in [50,100,200]:
        root=Path('results/eeg_latent' if span==50 else f'results/eeg_latent_span{span}')
        stats=json.loads((root/'control_diagnostics.json').read_text())
        for model in ['latent','latent_final']:
            score=json.loads((root/model/'classification.json').read_text())['test']
            rows.append({'span_ms':span,'model':model,**score,**stats[model]})
    frame=pd.DataFrame(rows);frame.to_csv('reports/eeg_latent_span_comparison.csv',index=False)
    print(frame.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
