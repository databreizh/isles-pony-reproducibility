"""Adapt DINOv2 ViT-S/14's last block and final norm on the original partitions."""
import argparse,copy,hashlib,json,os,platform,time
from pathlib import Path
os.environ.setdefault('XFORMERS_DISABLED','1')
import numpy as np
from common import manifests,fingerprint,load_embeddings,validation_score,evaluate,write_csv
from evaluation import load_image,file_hash

def main(a):
    import torch
    from torchvision import transforms
    torch.set_num_threads(a.threads)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    ms,cat=manifests(a.splits);ids,reference=load_embeddings(a.embeddings,cat);lookup={i:n for n,i in enumerate(ids)}
    a.output.mkdir(parents=True,exist_ok=True)
    config=vars(a).copy();config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    config.update(device_used=device,torch_version=torch.__version__,python=platform.python_version(),
                  adaptation='last transformer block + final norm; no added projection; normalized 384D CLS',
                  augmentation=False,optimizer='Adam coupled weight decay',margin=.2,identities_per_batch=16,images_per_identity=4)
    (a.output/'run_config.json').write_text(json.dumps(config,indent=2))
    paths=[]
    for i in ids:
        path=a.base.joinpath(*cat[i]['relative_path'].split('/'))
        if not path.is_file() or file_hash(path)!=cat[i]['sha256']:raise ValueError(f'Missing or changed image: {cat[i]["relative_path"]}')
        paths.append(path)
    torch.manual_seed(42)
    model=torch.hub.load('facebookresearch/dinov2:main','dinov2_vits14',trust_repo=True).eval().to(device)
    if len(model.blocks)!=12:raise ValueError('Expected unchunked ViT-S/14 with 12 blocks')
    model_hash=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        model_hash.update(name.encode());model_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    weight_hash=model_hash.hexdigest();config['initial_model_sha256']=weight_hash
    for p in model.parameters():p.requires_grad_(False)
    for module in [model.blocks[-1],model.norm]:
        for p in module.parameters():p.requires_grad_(True)
    params=[p for p in model.parameters() if p.requires_grad]
    config['trainable_parameters']=sum(p.numel() for p in params)
    config['total_parameters']=sum(p.numel() for p in model.parameters())
    initial_block=copy.deepcopy(model.blocks[-1].state_dict());initial_norm=copy.deepcopy(model.norm.state_dict())
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    cache=a.output/'prefix_tokens.npy';meta=a.output/'prefix_tokens_metadata.json'
    expected={'catalog_fingerprint':fingerprint(cat),'weights_sha256':weight_hash,'preprocess':'exif_rgb_square_gray127_resize224_bicubic_v1'}
    if cache.exists() and meta.exists():
        if json.loads(meta.read_text())!=expected:raise ValueError('Prefix cache mismatch; use a new output folder')
        tokens=np.load(cache,mmap_mode='r')
    else:
        with torch.no_grad():
            tokens=None
            for start in range(0,len(ids),a.batch_size):
                batch=torch.stack([transform(load_image(p)) for p in paths[start:start+a.batch_size]]).to(device)
                x=model.prepare_tokens_with_masks(batch)
                for block in model.blocks[:-1]:x=block(x)
                arr=x.cpu().numpy().astype(np.float32)
                if tokens is None:tokens=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=(len(ids),*arr.shape[1:]))
                tokens[start:start+len(arr)]=arr
                print('Prefix extraction',min(start+len(arr),len(ids)),'/',len(ids),flush=True)
        tokens.flush();meta.write_text(json.dumps(expected,indent=2));tokens=np.load(cache,mmap_mode='r')
    if tokens.shape!=(len(ids),257,384) or tokens.dtype!=np.float32:raise ValueError('Invalid token cache')
    def forward(indices):
        x=torch.from_numpy(np.array(tokens[indices],copy=True)).to(device)
        x=model.blocks[-1](x);x=model.norm(x)[:,0]
        return torch.nn.functional.normalize(x,dim=-1)
    def extract(indices):
        result=np.zeros((len(ids),384),dtype=np.float64)
        with torch.no_grad():
            for start in range(0,len(indices),a.batch_size):
                ix=indices[start:start+a.batch_size];result[ix]=forward(ix).cpu().numpy()
        return result
    # Ensure the new implementation and pretrained weights reproduce the frozen control.
    raw=extract(list(range(len(ids))))
    error=float(np.max(np.abs(raw-reference)));config['max_abs_difference_from_frozen_cache']=error
    if error>2e-5:raise ValueError(f'Frozen features disagree with existing cache ({error}); resolve weights/preprocessing before training')
    (a.output/'run_config.json').write_text(json.dumps(config,indent=2))
    predictions=[];results=[];selection=[];history=[]
    for m in ms:
        seed=m['seed'];seedout=a.output/f'seed_{seed}';seedout.mkdir(exist_ok=True)
        if (seedout/'complete.json').exists():
            saved=json.loads((seedout/'complete.json').read_text())
            if saved['run_config']!=config:raise ValueError('Completed seed has different settings; choose new output folder')
            results+=saved['results'];predictions+=saved['predictions'];selection.append(saved['selection']);history+=saved['history'];continue
        model.blocks[-1].load_state_dict(initial_block);model.norm.load_state_dict(initial_norm);model.eval()
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        rng=np.random.default_rng(seed);train=m['reserved_train_identities']
        groups=[[lookup[i] for i in ids if cat[i]['identity']==label] for label in train]
        valset=set(m['validation_identities']);valix=[lookup[i] for i in ids if cat[i]['identity'] in valset]
        bestscore=validation_score(m,raw,lookup);bestepoch=0
        bestblock=copy.deepcopy(initial_block);bestnorm=copy.deepcopy(initial_norm)
        optimizer=torch.optim.Adam(params,lr=a.learning_rate,weight_decay=1e-4)
        localhistory=[dict(seed=seed,epoch=0,training_loss='',validation_top1=bestscore)]
        started=time.perf_counter()
        for epoch in range(1,a.epochs+1):
            losses=[]
            for step in range(a.steps):
                labels=rng.choice(len(train),16,replace=False)
                ix=np.concatenate([rng.choice(groups[i],4,replace=False) for i in labels])
                y=torch.tensor(np.repeat(labels,4),device=device)
                optimizer.zero_grad(set_to_none=True)
                # Chunk forwards, but retain the complete 64-image graph for batch-hard mining.
                z=torch.cat([forward(ix[s:s+a.batch_size]) for s in range(0,len(ix),a.batch_size)])
                sim=z@z.T;same=y[:,None]==y[None,:];same.fill_diagonal_(False)
                pos=sim.masked_fill(~same,float('inf')).min(dim=1).values
                neg=sim.masked_fill(y[:,None]==y[None,:],-float('inf')).max(dim=1).values
                loss=(.2+neg-pos).clamp_min(0).mean()
                if not torch.isfinite(loss):raise ValueError('Non-finite loss')
                loss.backward();optimizer.step();losses.append(float(loss.detach()))
            if epoch%5==0 or epoch==a.epochs:
                f=extract(valix);score=validation_score(m,f,lookup)
                localhistory.append(dict(seed=seed,epoch=epoch,training_loss=float(np.mean(losses)),validation_top1=score))
                print(f'Seed {seed}, epoch {epoch}: validation={score:.4f}',flush=True)
                if score>bestscore:
                    bestscore=score;bestepoch=epoch;bestblock=copy.deepcopy(model.blocks[-1].state_dict());bestnorm=copy.deepcopy(model.norm.state_dict())
        elapsed=time.perf_counter()-started
        model.blocks[-1].load_state_dict(bestblock);model.norm.load_state_dict(bestnorm)
        f=extract(list(range(len(ids))))
        r,p=evaluate(m,f,ids,'dinov2_last_block');results+=r;predictions+=p
        sel=dict(seed=seed,selected_epoch=bestepoch,validation_score=bestscore,training_and_validation_seconds=elapsed,trainable_parameters=config['trainable_parameters'])
        selection.append(sel);history+=localhistory
        torch.save({'last_block':{k:v.cpu() for k,v in bestblock.items()},'norm':{k:v.cpu() for k,v in bestnorm.items()},
                    'initial_model_sha256':weight_hash,'seed':seed,'selected_epoch':bestepoch,'config':config},seedout/'selected_last_block.pt')
        np.savez_compressed(seedout/'embeddings_last_block.npz',features=f,ids=np.array(ids),fingerprint=fingerprint(cat))
        (seedout/'complete.json').write_text(json.dumps({'run_config':config,'selection':sel,'history':localhistory,'results':r,'predictions':p},indent=2))
    write_csv(a.output/'predictions_last_block.csv',predictions);write_csv(a.output/'training_history.csv',history);write_csv(a.output/'selection.csv',selection)
    (a.output/'results_last_block.json').write_text(json.dumps({'config':config,'selection':selection,'results':results},indent=2))
    print('Completed:',a.output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--splits',type=Path,default=Path(__file__).parent/'splits')
    p.add_argument('--output',type=Path,default=Path('last_block_results'))
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--steps',type=int,default=10)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    a=p.parse_args()
    if min(a.epochs,a.steps,a.batch_size,a.threads)<1 or a.learning_rate<=0:p.error('All numeric settings must be positive')
    main(a)
