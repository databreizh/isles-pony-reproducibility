"""Paired identity-cluster bootstrap and descriptive open-set tradeoff curves."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import write_csv

def boolean(v):
    if v.lower() in ['true','1']:return True
    if v.lower() in ['false','0']:return False
    raise ValueError(f'Invalid Boolean: {v}')

def load(paths):
    rows=[];seen=set()
    for path in paths:
        text=path.read_text(encoding='utf-8-sig');delimiter=';' if ';' in text.splitlines()[0] else ','
        for r in csv.DictReader(text.splitlines(),delimiter=delimiter):
            if not r.get('method'):raise ValueError('Every row needs a method column; use export_predictions.py')
            r['seed']=int(r['seed']);r['gallery_size']=int(r['gallery_size']);r['score']=float(r['score'])
            r['is_unknown']=boolean(r['is_unknown'])
            if r.get('accepted','')!='':r['accepted']=boolean(r['accepted'])
            key=tuple(r[k] for k in ['method','seed','protocol','gallery_size','image_index'])
            if key in seen:raise ValueError(f'Duplicate prediction: {key}')
            seen.add(key);rows.append(r)
    return rows

def paired_bootstrap(rows,a,b,k,n,seed):
    selected=[r for r in rows if r['protocol']=='reid_25' and r['gallery_size']==k and r['method'] in [a,b]]
    methods={name:{(r['seed'],r['image_index']):r for r in selected if r['method']==name} for name in [a,b]}
    if not methods[a] or methods[a].keys()!=methods[b].keys():raise ValueError('Methods must have exactly matching closed-set queries')
    buckets=defaultdict(list)
    for key,ra in methods[a].items():
        rb=methods[b][key]
        if ra['true_identity']!=rb['true_identity']:raise ValueError('Identity mismatch')
        buckets[(ra['seed'],ra['true_identity'])].append(float(rb['predicted_identity']==rb['true_identity'])-float(ra['predicted_identity']==ra['true_identity']))
    identities=sorted({i for _,i in buckets});seeds=sorted({s for s,_ in buckets});lookup={i:n for n,i in enumerate(identities)}
    arrays=[]
    for s in seeds:
        groups=[(lookup[i],float(np.mean(v))) for (ss,i),v in buckets.items() if ss==s]
        arrays.append((np.array([i for i,v in groups]),np.array([v for i,v in groups])))
    point=float(np.mean([v.mean() for _,v in arrays]));rng=np.random.default_rng(seed);boot=[]
    while len(boot)<n:
        weights=np.bincount(rng.integers(0,len(identities),len(identities)),minlength=len(identities))
        if any(weights[ix].sum()==0 for ix,_ in arrays):continue
        boot.append(np.mean([np.average(v,weights=weights[ix]) for ix,v in arrays]))
    lo,hi=np.quantile(boot,[.025,.975])
    return {'method_a':a,'method_b':b,'difference_b_minus_a_pp':point*100,'ci95_low_pp':lo*100,'ci95_high_pp':hi*100,
            'unique_test_identities':len(identities),'partitions':len(seeds),'paired_query_occurrences':len(methods[a]),
            'replicates':n,'estimand':'mean of partition macro Top-1 differences',
            'scope':'Identity-cluster percentile interval conditional on recorded fitted models and episodes; repeated identity occurrences receive the same bootstrap multiplicity across all partitions. No retraining, session correction, or confirmatory p-value.'}

def macro(rows,values):
    groups=defaultdict(list)
    for r,v in zip(rows,values):groups[r['true_identity']].append(v)
    if not groups:raise ValueError('Empty identity cohort')
    return float(np.mean([np.mean(v) for v in groups.values()]))

def rates(known,unknown,t):
    ka=np.array([r['score']>=t for r in known]);ua=np.array([r['score']>=t for r in unknown])
    correct=np.array([r['true_identity']==r['predicted_identity'] for r in known])
    return {'cia':macro(known,ka&correct),'known_rejection':macro(known,~ka),
            'known_wrong_acceptance':macro(known,ka&~correct),'ufa':macro(unknown,ua)}

def curves(rows,k,out):
    groups=defaultdict(list)
    for r in rows:
        if r['protocol']=='open_test' and r['gallery_size']==k:groups[(r['method'],r['seed'])].append(r)
    if not groups:raise ValueError('No open_test rows')
    curve=[];operating=[];interpolated=defaultdict(list);grid=np.linspace(0,1,201)
    for (method,seed),rs in sorted(groups.items()):
        known=[r for r in rs if not r['is_unknown']];unknown=[r for r in rs if r['is_unknown']]
        scores=sorted({r['score'] for r in rs},reverse=True)
        thresholds=[float(np.nextafter(scores[0],np.inf))]+scores
        points=[]
        for t in thresholds:
            result=rates(known,unknown,t);points.append(result);curve.append(dict(method=method,seed=seed,threshold=t,**result))
        # Best attainable CIA with UFA <= each target, descriptive TEST envelope.
        interpolated[method].append([max(p['cia'] for p in points if p['ufa']<=u+1e-12) for u in grid])
        if all(isinstance(r.get('accepted'),bool) for r in rs):
            ka=np.array([r['accepted'] for r in known]);ua=np.array([r['accepted'] for r in unknown]);correct=np.array([r['true_identity']==r['predicted_identity'] for r in known])
            operating.append({'method':method,'seed':seed,'cia':macro(known,ka&correct),'known_rejection':macro(known,~ka),
                              'known_wrong_acceptance':macro(known,ka&~correct),'ufa':macro(unknown,ua)})
    write_csv(out/'open_set_curves_by_seed.csv',curve)
    if operating:write_csv(out/'open_set_operating_errors.csv',operating)
    average=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for method,v in sorted(interpolated.items()):
        v=np.asarray(v);mean=v.mean(axis=0);sd=v.std(axis=0,ddof=1) if len(v)>1 else np.zeros(len(grid))
        average.extend({'method':method,'ufa_budget':u,'mean_cia':c,'partition_sd':s} for u,c,s in zip(grid,mean,sd))
        ax.step(grid*100,mean*100,where='post',label=method)
        ops=[r for r in operating if r['method']==method]
        if ops:ax.scatter(np.mean([r['ufa'] for r in ops])*100,np.mean([r['cia'] for r in ops])*100,s=25)
    ax.set(
    xlabel="UFA budget (%)",
    ylabel="CIA (%)",
    xlim=(0, 20),
    ylim=(0, 100),
    )
    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'cia_ufa.pdf');fig.savefig(out/'cia_ufa.png',dpi=180);plt.close(fig)
    write_csv(out/'open_set_mean_envelope.csv',average)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,nargs='+',required=True)
    p.add_argument('--method-a',default='dinov2');p.add_argument('--method-b',default='projection_triplet')
    p.add_argument('--gallery-size',type=int,choices=[1,3,5],default=5)
    p.add_argument('--replicates',type=int,default=10000);p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--output',type=Path,default=Path('analysis_results'))
    a=p.parse_args()
    if a.replicates<100 or a.method_a==a.method_b:p.error('Use >=100 replicates and two different methods')
    a.output.mkdir(parents=True,exist_ok=True);rows=load(a.predictions)
    result=paired_bootstrap(rows,a.method_a,a.method_b,a.gallery_size,a.replicates,a.seed)
    (a.output/'paired_bootstrap.json').write_text(json.dumps(result,indent=2));curves(rows,a.gallery_size,a.output)
    (a.output/'interpretation.txt').write_text('Bootstrap: conditional identity-cluster uncertainty; not training uncertainty and not a correction for intra-session dependence.\nCurves: retrospective test-score envelopes, not validation-calibrated deployment thresholds. Dots are the existing validation-calibrated operating points. Do not select an operational threshold from these curves. Partition SD is descriptive.\n')
    print(json.dumps(result,indent=2))
