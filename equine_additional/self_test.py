"""Small checks for paired resampling and open-set error accounting (no GPU)."""
from analyze_predictions import paired_bootstrap,rates

def main():
    rows=[]
    for seed,identities in [(42,['pony_a','pony_b']),(43,['pony_b','pony_c'])]:
        for i in identities:
            for method,pred in [('raw','wrong'),('learned',i)]:
                rows.append(dict(seed=seed,protocol='reid_25',gallery_size=5,method=method,
                                 image_index=i,true_identity=i,predicted_identity=pred))
    result=paired_bootstrap(rows,'raw','learned',5,500,2027)
    assert result['unique_test_identities']==3
    assert result['difference_b_minus_a_pp']==100
    assert result['ci95_low_pp']==100 and result['ci95_high_pp']==100
    reverse=paired_bootstrap(rows,'learned','raw',5,500,2027)
    assert reverse['difference_b_minus_a_pp']==-100
    known=[dict(true_identity='a',predicted_identity='a',score=.9),
           dict(true_identity='b',predicted_identity='a',score=.8),
           dict(true_identity='c',predicted_identity='c',score=.2)]
    unknown=[dict(true_identity='u',score=.7),dict(true_identity='v',score=.1)]
    r=rates(known,unknown,.5)
    assert abs(r['cia']-1/3)<1e-12 and abs(r['known_rejection']-1/3)<1e-12
    assert abs(r['known_wrong_acceptance']-1/3)<1e-12 and r['ufa']==.5
    assert abs(r['cia']+r['known_rejection']+r['known_wrong_acceptance']-1)<1e-12
    broken=[r for r in rows if not(r['method']=='learned' and r['seed']==43 and r['image_index']=='pony_c')]
    try:paired_bootstrap(broken,'raw','learned',5,500,2027)
    except ValueError:pass
    else:raise AssertionError('Unpaired queries were not rejected')
    print('Identity overlap, paired differences, unmatched-query rejection and error decomposition: OK')

if __name__=='__main__':main()
