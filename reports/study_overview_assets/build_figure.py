"""Editable study overview from real saved study signals (no raster SVG content).

Run from the repository root:
    .venv-gpu/bin/python reports/study_overview_assets/build_figure.py

Uses an archived numerical extract once present, so the SVG can be rebuilt without
large/private cache paths. Inkscape is used only to export the PNG preview.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SVG = ROOT/'reports/study_overview.svg'
PNG = ROOT/'reports/study_overview.png'
NS = 'http://www.w3.org/2000/svg'
# Official Inkscape namespace; each stage is a named editable layer.
INK = 'http://www.inkscape.org/namespaces/inkscape'
ET.register_namespace('', NS)
ET.register_namespace('inkscape', INK)
BLUE, ORANGE = '#246DA4', '#C66A20'
INK_COLOR, GRAY, LIGHT = '#25313B', '#66717B', '#D6DCE1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    snapshot = HERE/'signal_extract.npz'
    provenance = HERE/'signal_provenance.json'
    if not snapshot.exists():
        sys.path.insert(0, str(ROOT))
        from temporal_ssl.data import simulate
        source = ROOT/'results/research/predictable-causal-d7-m11'
        record = json.loads((source/'split_manifest.json').read_text())['test'][0]
        sim = simulate(record['simulation'], np.random.SeedSequence(
            record['seed_entropy'], spawn_key=tuple(record['spawn_key'])))
        with np.load(source/'reconstruction.npz') as r, np.load(source/'normalization.npz') as n:
            # Exact identity with the actual stored example, not a new realization.
            np.testing.assert_array_equal((sim['x']-n['mean'])/n['std'], r['x'])
        subject = json.loads((ROOT/'results/eeg_pilot/participants.json').read_text())['train'][0]
        channels = [0, 2, 4, 6]  # Fp1, F3, C3, P3 in the fixed study montage.
        with np.load(subject['cache']) as r:
            eeg = r['x'][0, :1000][:, channels].copy()
            source_start = int(r['source_start'][0])
        np.savez(snapshot, synthetic=sim['x'], u=sim['latents']['u'][:, 0],
                 q=sim['latents']['q'][:, 0], phase=sim['latents']['phi'][:, 0],
                 eeg=eeg, fs=np.array(500.))
        provenance.write_text(json.dumps({
            'synthetic': {'source':str(source.relative_to(ROOT)), 'split':'test', 'trajectory_index':0,
                'seed_entropy':record['seed_entropy'], 'spawn_key':record['spawn_key'],
                'parameters':sim['parameters'], 'verification':'exact equality to saved normalized reconstruction x',
                'manifest_sha256':sha(source/'split_manifest.json')},
            'eeg': {'dataset':'ds004504', 'participant':subject['participant_id'], 'split':'train',
                'epoch_index':0, 'source_start_sample':source_start, 'display_samples':[0,1000],
                'channels':['Fp1','F3','C3','P3'], 'sampling_hz':500,
                'cache_sha256':sha(subject['cache']), 'units':'microvolts',
                'processing':'cached cleaned 0.5–45 Hz derivatives; no new filtering or per-channel scaling'},
            'display': {'eeg':'constant vertical offsets only; common 0.12 viewBox units per microvolt',
                'synthetic':'first observed channel and true accumulator/regime; separate labeled lanes',
                'patches':'schematic counts; visible waveform snippets are actual 25-sample EEG patches; hidden slots contain no fabricated predictions'},
            'signal_extract_sha256':sha(snapshot)}, indent=2)+'\n')
    return dict(np.load(snapshot)), json.loads(provenance.read_text())


def element(parent, tag, **attributes):
    return ET.SubElement(parent, f'{{{NS}}}{tag}', {k.replace('_','-'):str(v) for k,v in attributes.items()})


def text(parent, x, y, content, size=17, fill=INK_COLOR, weight='normal', anchor='start', **kw):
    node = element(parent, 'text', x=x, y=y, fill=fill, font_size=size,
                   font_family='Arial, Helvetica, sans-serif', font_weight=weight, text_anchor=anchor, **kw)
    node.text = content
    return node


def line(parent, x1, y1, x2, y2, color=LIGHT, width=1.2, **kw):
    return element(parent, 'line', x1=x1,y1=y1,x2=x2,y2=y2,stroke=color,stroke_width=width,**kw)


def path(parent, points, color, width=1.5, **kw):
    return element(parent, 'path', d='M '+' L '.join(f'{x:.3f},{y:.3f}' for x,y in points),
                   fill='none',stroke=color,stroke_width=width,stroke_linecap='round',stroke_linejoin='round',**kw)


def rect(parent,x,y,w,h,stroke=LIGHT,fill='white',radius=0,**kw):
    return element(parent,'rect',x=x,y=y,width=w,height=h,stroke=stroke,fill=fill,rx=radius,**kw)


def arrow(parent,x1,y1,x2,y2,color=GRAY,width=1.3):
    line(parent,x1,y1,x2,y2,color,width)
    angle=np.arctan2(y2-y1,x2-x1)
    d=np.array([np.cos(angle),np.sin(angle)])
    n=np.array([-d[1],d[0]])
    tip=np.array([x2,y2]);base=tip-6*d
    pts=[tip,base+2.6*n,base-2.6*n,tip]
    return element(parent,'path',d='M '+' L '.join(f'{x:.2f},{y:.2f}' for x,y in pts),fill=color,stroke='none')


def stage(parent,n,label,y):
    text(parent,30,y,str(n),size=18,weight='bold',fill=GRAY)
    text(parent,57,y,label.upper(),size=17,weight='bold',fill=GRAY,letter_spacing='1.2')


def layer(parent,label):
    node=element(parent,'g',id=label.lower().replace(' ','-'))
    node.set(f'{{{INK}}}groupmode','layer');node.set(f'{{{INK}}}label',label)
    return node


def bracket(parent,x1,x2,y,label,color):
    line(parent,x1,y,x2,y,color,1)
    line(parent,x1,y-4,x1,y,color,1);line(parent,x2,y-4,x2,y,color,1)
    text(parent,(x1+x2)/2,y+20,label,size=15,fill=color,anchor='middle')


def patches(parent,x,y,color,signal,masked):
    w,h,gap=43,36,6
    for i in range(8):
        xx=x+i*(w+gap)
        hidden=i in masked
        rect(parent,xx,y,w,h,stroke=color,fill='#F5F6F7' if hidden else 'white',
             stroke_width=1.25,stroke_dasharray='3 2' if hidden else 'none')
        if hidden:
            # Hidden means unknown, rather than an invented waveform prediction.
            text(parent,xx+w/2,y+24,'?',size=20,fill=color,anchor='middle')
        else:
            values=signal[i*25:(i+1)*25]
            points=zip(np.linspace(xx+3,xx+w-3,25),y+h/2-values*.45)
            path(parent,points,color,1.1)
    return 8*w+7*gap


def embeddings(parent,x,y,color):
    # Vector glyphs deliberately abstract, with no quantitative heatmap implied.
    for i in range(8):
        xx=x+27*i
        rect(parent,xx,y,20,30,stroke=color,fill='white',stroke_width=1)
        for j in range(1,4):
            line(parent,xx+4,y+j*6,xx+16,y+j*6,LIGHT,1)


def build():
    data, provenance=load_data()
    svg=ET.Element(f'{{{NS}}}svg',{'width':'170mm','height':'139.4mm','viewBox':'0 0 1000 820',
         'version':'1.1','role':'img','aria-labelledby':'figure-title figure-description'})
    element(svg,'title',id='figure-title').text='Study overview of causal forecasting and bidirectional masked reconstruction'
    element(svg,'desc',id='figure-description').text=(
        'Four stages: actual predictable-oscillator and multichannel EEG data; separate Transformer waveform objectives; '
        'frozen patch embeddings; waveform prediction through reconstruction heads and frozen readouts for content and diagnosis. '
        'Waveform evaluation applies to both datasets, latent recovery to synthetic data, spectral accessibility and diagnosis to EEG. '
        'All signal traces come from the archived actual study examples. Patch counts are schematic.')
    element(svg,'metadata').text=json.dumps(provenance,ensure_ascii=False)
    rect(svg,0,0,1000,820,stroke='none',fill='white')
    g=layer(svg,'1 Datasets')
    stage(g,1,'Datasets',30)
    text(g,60,63,'Synthetic dynamical systems',size=20,weight='bold')
    text(g,550,63,'Resting-state EEG',size=20,weight='bold')
    text(g,60,86,'Predictable switching oscillator',size=16,fill=GRAY)
    text(g,550,86,'ds004504 · 19 channels · 500 Hz',size=16,fill=GRAY)
    # Synthetic signal and true factors, all on their original time index.
    x=np.linspace(98,467,128)
    signal=data['synthetic'][:,0]
    q=data['q'];u=data['u']
    switches=np.flatnonzero(np.diff(q)!=0)+1
    for s in switches:
        xx=x[s];line(g,xx,103,xx,192,'#C9CFD4',1,stroke_dasharray='2 3')
    text(g,60,127,'x₁',size=15,fill=GRAY)
    path(g,zip(x,124-17*signal),INK_COLOR,1.35)
    text(g,60,167,'u',size=15,fill=GRAY)
    path(g,zip(x,180-28*u),GRAY,1.35)
    text(g,60,199,'q',size=15,fill=GRAY)
    edges=np.r_[0,switches,128]
    for start,stop in zip(edges[:-1],edges[1:]):
        left=98+369*start/128;right=98+369*stop/128
        rect(g,left,190,right-left,10,stroke='none',fill='#A1ABB4' if q[start] else '#E7EAED')
    text(g,98,219,'0',size=13,fill=GRAY)
    text(g,467,219,'128 samples',size=13,fill=GRAY,anchor='end')
    text(g,60,251,'Known factors: phase, accumulator, regime',size=16,weight='bold')
    # EEG common physical voltage scale, no channel-specific normalization.
    eeg=data['eeg'];tx=602+np.arange(len(eeg))*326/(2*float(data['fs']))
    for i,name in enumerate(['Fp1','F3','C3','P3']):
        y=111+i*27
        text(g,550,y+5,name,size=14,fill=GRAY)
        path(g,zip(tx,y-.12*eeg[:,i]),INK_COLOR,1.05)
    line(g,602,219,683.5,219,GRAY,1.2)
    line(g,602,215,602,223,GRAY,1);line(g,683.5,215,683.5,223,GRAY,1)
    text(g,642.75,232,'0.5 s',size=13,anchor='middle',fill=GRAY)
    line(g,850,214,850,226,GRAY,1.2)
    line(g,846,214,854,214,GRAY,1);line(g,846,226,854,226,GRAY,1)
    text(g,864,225,'100 µV',size=13,fill=GRAY)
    text(g,550,251,'Physiological generative factors unknown',size=16,weight='bold')
    line(g,30,267,970,267,LIGHT,.9)

    g=layer(svg,'2 Self supervised pretraining')
    stage(g,2,'Self-supervised pretraining',294)
    text(g,60,325,'Causal forecasting',size=20,weight='bold',fill=BLUE)
    text(g,550,325,'Bidirectional masked reconstruction',size=20,weight='bold',fill=ORANGE)
    snippets=eeg[:200,1]
    patches(g,60,341,BLUE,snippets,set(range(4,8)))
    patches(g,550,341,ORANGE,snippets,{2,3,6})
    bracket(g,60,250,385,'Observed context',BLUE)
    bracket(g,256,446,385,'Future targets',BLUE)
    bracket(g,550,936,385,'Visible context on both sides of gaps',ORANGE)
    # Directional access glyphs: causal past→future; masked context→hidden.
    arrow(g,170,416,335,416,BLUE)
    arrow(g,590,416,692,416,ORANGE)
    arrow(g,809,416,728,416,ORANGE)
    arrow(g,928,416,870,416,ORANGE)
    # Independent encoders, not shared weights between the two objectives.
    for xx,color in [(60,BLUE),(550,ORANGE)]:
        rect(g,xx,433,238,33,stroke=color,stroke_width=1.1,radius=3)
        text(g,xx+119,455,'Transformer encoder',size=17,anchor='middle')
        arrow(g,xx+244,449,xx+260,449,color)
        rect(g,xx+268,433,118,33,stroke=color,stroke_width=1.1,radius=3)
        text(g,xx+327,455,'Waveform head',size=14.5,anchor='middle')
    line(g,30,487,970,487,LIGHT,.9)

    g=layer(svg,'3 Frozen representations')
    stage(g,3,'Learned representations',514)
    for xx,color in [(60,BLUE),(550,ORANGE)]:
        arrow(g,xx+119,468,xx+119,481,color)
        text(g,xx+5,547,'Frozen encoder',size=17,weight='bold',fill=color)
        # Small lock glyph using editable vector paths.
        rect(g,xx+158,536,12,11,stroke=GRAY,stroke_width=1,fill='white',radius=1)
        element(g,'path',d=f'M {xx+161},536 v -4 a 3,3 0 0 1 6,0 v 4',fill='none',stroke=GRAY,stroke_width=1)
        embeddings(g,xx+5,561,color)
        text(g,xx+235,574,'Temporal patch',size=16,fill=GRAY)
        text(g,xx+235,594,'embeddings',size=16,fill=GRAY)
    text(g,500,619,'Frozen features → simple readouts; untrained and handcrafted controls',
         size=16,anchor='middle',fill=GRAY)
    line(g,30,638,970,638,LIGHT,.9)

    g=layer(svg,'4 Complementary evaluations')
    stage(g,4,'Complementary evaluations',666)
    # No global arrow from frozen embeddings to waveform MSE: that evaluation uses heads.
    for xx,title,tag in [(60,'Predictive performance','SYNTHETIC + EEG'),
                         (380,'Representation content','SYNTHETIC + EEG'),
                         (720,'Biomedical utility','EEG ONLY')]:
        text(g,xx,696,tag,size=13,weight='bold',fill=GRAY,letter_spacing='.7')
        text(g,xx,723,title,size=18.5,weight='bold')
    line(g,356,691,356,800,LIGHT,1)
    line(g,697,691,697,800,LIGHT,1)
    text(g,60,749,'Waveform head → MSE',size=17)
    text(g,60,775,'vs persistence / interpolation',size=16,fill=GRAY)
    text(g,60,799,'Both objectives, own targets',size=15,fill=GRAY)
    text(g,380,749,'Synthetic: latent-factor recovery',size=15.5)
    text(g,380,775,'EEG: spectral accessibility',size=16)
    text(g,380,799,'Frozen linear probes · R² / BA',size=15,fill=GRAY)
    text(g,720,749,'Dementia classification',size=17)
    text(g,720,775,'AD / FTD / healthy controls',size=15.5)
    text(g,720,799,'Frozen features · BA / macro-F1',size=14.5,fill=GRAY)
    ET.indent(svg)
    ET.ElementTree(svg).write(SVG,encoding='utf-8',xml_declaration=True)
    subprocess.run(['inkscape',str(SVG),'--export-type=png',f'--export-filename={PNG}',
                    '--export-dpi=600','--export-background=white'],check=True,capture_output=True)
    print(SVG);print(PNG)


if __name__=='__main__':
    build()
