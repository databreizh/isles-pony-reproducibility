"""Plot validation histories and validation-calibrated open-set operating points."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter=';'))

def make_plots(root):
    output=root/'figures';output.mkdir(exist_ok=True)
    for folder in sorted({p.parent for p in root.rglob('training_history_seed_*.csv')}):
        fig,ax=plt.subplots(figsize=(6,4))
        for p in sorted(folder.glob('training_history_seed_*.csv')):
            rows=read(p);seed=rows[0]['seed']
            ax.plot([int(r['epoch']) for r in rows],[float(r['validation_top1_mean_1_3_5'])*100 for r in rows],label=f'seed {seed}')
        ax.set(xlabel='Époque',ylabel='Validation macro Top-1 moyenne 1/3/5 (%)',title=folder.name)
        ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(output/(folder.name+'_validation.png'),dpi=180);plt.close(fig)
    for stage in ['projections','background']:
        path=root/stage/'operating_points.csv'
        if not path.exists():continue
        groups=defaultdict(list)
        for r in read(path):
            if r['gallery_size']=='5':
                groups[(r['method'],r['condition'],float(r['validation_target_photo_ufa']))].append(r)
        # Separate figures avoid comparing different candidate sets/cohorts.
        families=sorted({method.rsplit('_',1)[0] for method,condition,alpha in groups}) if stage=='projections' else ['background']
        for family in families:
            series=defaultdict(list)
            for (method,condition,alpha),rows in groups.items():
                if stage=='projections' and not method.startswith(family+'_'):continue
                series[(method,condition)].append((alpha,np.mean([float(r['test_ufa_macro'])*100 for r in rows]),
                    np.mean([float(r['test_cia_macro'])*100 for r in rows])))
            fig,ax=plt.subplots(figsize=(7,4.6))
            for (method,condition),points in sorted(series.items()):
                points=sorted(points);ax.plot([p[1] for p in points],[p[2] for p in points],marker='.',label=method+' / '+condition)
            ax.set(xlabel='Test UFA macro (%)',ylabel='Test CIA macro (%)',title='Points calibrés sur validation, 5 références')
            ax.grid(alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=6.5)
            fig.tight_layout();fig.savefig(output/(family+'_open_set.png'),dpi=180,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path(__file__).parent/'results')
    make_plots(p.parse_args().results)
