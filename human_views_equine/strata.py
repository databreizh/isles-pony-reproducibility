from collections import defaultdict
import evaluation as ev

def measure(m,ids,x,query_ids,gallery,method,condition,analysis,reference_view,query_view,k):
    cat=m['catalog'];query_ids=list(map(str,query_ids));gallery=list(map(str,gallery))
    n=len({cat[i]['identity'] for i in gallery});nq=len({cat[i]['identity'] for i in query_ids})
    row={'seed':m['seed'],'method':method,'condition':condition,'analysis':analysis,
         'reference_view':reference_view,'query_view':query_view,'gallery_size':k,
         'candidate_identities':n,'query_identities':nq,'queries':len(query_ids),'gallery_images':len(gallery),
         'chance_top1':1/n if n else '', 'status':'ok' if n>=2 and query_ids else 'insufficient_data'}
    if row['status']!='ok':return row
    if not {cat[i]['identity'] for i in query_ids}<={cat[i]['identity'] for i in gallery}:
        raise ValueError('Identité requête absente des références')
    episode={'gallery':{str(k):gallery},'queries':query_ids}
    result=ev.evaluate_closed(episode,k,x,{str(i):p for p,i in enumerate(ids)},cat,m['seed'],[],'view_analysis')
    for name in ['top1_macro','top3_macro','top5_macro','top1_micro']:row[name]=result[name]
    return row

def paired_episodes(m,labels):
    """One head ref and one body ref, drawn ONLY from the held-out gallery 5.

    All four pair cells share the same candidate identities. Query-view cells
    can nevertheless differ in which identities have queries; report counts.
    """
    cat=m['catalog'];buckets=defaultdict(lambda:defaultdict(list))
    for i in m['reid_25']['gallery']['5']:
        key=str(i);buckets[cat[key]['identity']][labels[key]['view']].append(key)
    shared=sorted(label for label,v in buckets.items() if v['head'] and v['body'])
    galleries={view:[buckets[label][view][0] for label in shared] for view in ['head','body']}
    queries={view:[str(i) for i in m['reid_25']['queries'] if cat[str(i)]['identity'] in shared and labels[str(i)]['view']==view] for view in ['head','body']}
    return galleries,queries
