#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess, time
from pathlib import Path
import geopandas as gpd, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from rasterio.features import rasterize
from geotessera import GeoTesseraZarr
from geotessera.registry import zarr_store_url

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def gitsha(p):
    return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()

def procrustes(x_new,x_old):
    xm,ym=x_new.mean(0),x_old.mean(0)
    u,s,vt=np.linalg.svd((x_new-xm).T@(x_old-ym),full_matrices=False)
    return xm,ym,u@vt,s

def apply_map(x,xm,ym,t): return (x-xm)@t+ym

def ece(prob,y,bins=10):
    c=prob.max(1); ok=prob.argmax(1)==y; out=0.0
    for lo,hi in zip(np.linspace(0,1,bins)[:-1],np.linspace(0,1,bins)[1:]):
        m=(c>lo)&(c<=hi)
        if m.any(): out+=m.mean()*abs(c[m].mean()-ok[m].mean())
    return float(out)

def metrics(y,p,prob,ref):
    labs=np.unique(y)
    return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),
    'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
    'weighted_f1':float(f1_score(y,p,average='weighted',zero_division=0)),
    'ece10':ece(prob,y),'decision_churn_vs_old_features':float(np.mean(p!=ref)),
    'per_class_f1':{str(k):float(v) for k,v in zip(labs,f1_score(y,p,average=None,labels=labs,zero_division=0))}}

def boot(y,preds,seed,reps):
    rng=np.random.default_rng(seed); n=len(y); ans={k:[] for k in preds}
    for _ in range(reps):
        i=rng.integers(0,n,n)
        for k,p in preds.items(): ans[k].append(float(f1_score(y[i],p[i],average='macro',zero_division=0)))
    return {k:[float(np.mean(v)),float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in ans.items()}

def sample_polygon_means(gdf,version,year):
    c=GeoTesseraZarr(zarr_store_url(version)); t=time.time()
    mosaic,transform,crs=c.read_region(tuple(gdf.total_bounds),year=year,progress=False)
    gp=gdf.to_crs(crs)
    label_raster=rasterize(((geom,i+1) for i,geom in enumerate(gp.geometry)),out_shape=mosaic.shape[:2],
                           transform=transform,fill=0,dtype='int32',all_touched=True)
    x=np.full((len(gdf),mosaic.shape[2]),np.nan,dtype=np.float32); counts=np.zeros(len(gdf),dtype=np.int32)
    for i in range(len(gdf)):
        vals=mosaic[label_raster==i+1]; valid=np.isfinite(vals).all(1)
        counts[i]=int(valid.sum())
        if valid.any(): x[i]=vals[valid].mean(0)
    return x,{'requested_version':version,'store_url':zarr_store_url(version),'dataset_version':c.dataset.version,
    'dataset_variant':c.dataset.variant,'seconds':time.time()-t,'mosaic_shape':list(mosaic.shape),'crs':str(crs),
    'labelled_pixels_min':int(counts.min()),'labelled_pixels_median':float(np.median(counts)),
    'labelled_pixels_total':int(counts.sum()),'finite_rows':int(np.isfinite(x).all(1).sum()),
    'nonfinite_rows':int((~np.isfinite(x).all(1)).sum())},counts

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--year',type=int,default=2024); ap.add_argument('--old-version',default='v1'); ap.add_argument('--new-version',default='v1.1')
    ap.add_argument('--ridge-alpha',type=float,default=1.0); ap.add_argument('--seed',type=int,default=17); ap.add_argument('--bootstrap',type=int,default=500)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    g=gpd.read_file(a.labels)
    if len(g)>500: raise ValueError('pilot cap exceeded')
    rp=g.geometry.representative_point(); g=g.assign(lon=rp.x,lat=rp.y).copy(); g['site_id']=[f'site-{i:04d}' for i in range(len(g))]
    med=float(g.lon.median()); g['split']=np.where(g.lon<med,'train_west','test_east')
    enc=LabelEncoder().fit(sorted(g.crop.unique())); y=enc.transform(g.crop.astype(str))
    old,om,old_counts=sample_polygon_means(g,a.old_version,a.year); new,nm,new_counts=sample_polygon_means(g,a.new_version,a.year)
    g['old_labelled_pixels']=old_counts; g['new_labelled_pixels']=new_counts
    g[['site_id','crop','lon','lat','split','old_labelled_pixels','new_labelled_pixels']].to_csv(a.out/'sample_manifest.csv',index=False)
    finite=np.isfinite(old).all(1)&np.isfinite(new).all(1)
    g.loc[~finite,['site_id','crop','lon','lat','split']].to_csv(a.out/'dropped_sites.csv',index=False)
    g=g[finite].reset_index(drop=True); old=old[finite]; new=new[finite]; y=y[finite]
    tr=(g.split=='train_west').to_numpy(); te=~tr
    xm,ym,t,sv=procrustes(new[tr],old[tr]); ridge=Ridge(alpha=a.ridge_alpha,random_state=a.seed).fit(new[tr],old[tr])
    variants={'old_features':old,'new_unmapped':new,'new_procrustes':apply_map(new,xm,ym,t),'new_ridge':ridge.predict(new)}
    heads={'logreg_balanced':LogisticRegression(max_iter=5000,class_weight='balanced',random_state=a.seed),
           'rf_balanced':RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',random_state=a.seed,n_jobs=-1)}
    res={'task':'austria_crops_polygon_mean_feature_migration','year':a.year,'n_sites':len(g),'n_train':int(tr.sum()),'n_test':int(te.sum()),
         'split_rule':'representative-point longitude below/above median','classes':list(enc.classes_),'old':om,'new':nm,
         'alignment_train_rows':int(tr.sum()),'alignment_rank':int(np.linalg.matrix_rank(new[tr]-new[tr].mean(0))),
         'procrustes_singular_values_top16':[float(v) for v in sv[:16]],'ridge_alpha':a.ridge_alpha,'runs':{}}
    predstore={}
    for hn,proto in heads.items():
        hold=proto.__class__(**proto.get_params()).fit(old[tr],y[tr]); ref=hold.predict(old[te]); runs={}
        for vn,x in variants.items():
            p=hold.predict(x[te]); prob=hold.predict_proba(x[te]); runs[vn]=metrics(y[te],p,prob,ref); predstore[f'{hn}/{vn}']=p
        hnew=proto.__class__(**proto.get_params()).fit(new[tr],y[tr]); p=hnew.predict(new[te]); prob=hnew.predict_proba(new[te])
        runs['new_head_refit']=metrics(y[te],p,prob,ref); predstore[f'{hn}/new_head_refit']=p; res['runs'][hn]=runs
    res['rf_macro_f1_bootstrap']=boot(y[te],{k:predstore[f'rf_balanced/{k}'] for k in variants.keys()|{'new_head_refit'}},a.seed,a.bootstrap)
    np.save(a.out/'features_old.npy',old); np.save(a.out/'features_new.npy',new); np.save(a.out/'labels.npy',y)
    res['provenance']={'label_path':str(a.labels),'label_sha256':sha256(a.labels),'geotessera_version':__import__('geotessera').__version__,
      'geotessera_git_sha':gitsha('/home/ubuntu/airr-masterplan/eo-task-research/geotessera-src'),
      'tessera_eval_git_sha':gitsha('/home/ubuntu/airr-masterplan/eo-task-research/tessera-eval-src'),'created_unix':int(time.time())}
    (a.out/'results.json').write_text(json.dumps(res,indent=2))
    lines=['# TESSERA compatibility pilot','',f'{len(g)} parcels; {tr.sum()} train west, {te.sum()} test east; year {a.year}.','Model+run migration falsifier, not an operational result.','',
           '| Head | Route | Macro-F1 | Balanced acc | Churn vs old features | ECE-10 |','|---|---|---:|---:|---:|---:|']
    for hn,runs in res['runs'].items():
        for vn,m in runs.items(): lines.append(f"| {hn} | {vn} | {m['macro_f1']:.3f} | {m['balanced_accuracy']:.3f} | {m['decision_churn_vs_old_features']:.3f} | {m['ece10']:.3f} |")
    lines+=['','RF macro-F1 bootstrap mean [2.5%,97.5%]:']
    for k,v in res['rf_macro_f1_bootstrap'].items(): lines.append(f'- {k}: {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]')
    (a.out/'report.md').write_text('\n'.join(lines)+'\n'); print(json.dumps({'out':str(a.out),'n':len(g),'train':int(tr.sum()),'test':int(te.sum())}))
if __name__=='__main__': main()
