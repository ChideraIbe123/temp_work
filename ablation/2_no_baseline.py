"""ABLATION: ML only, no Aug 2024 baseline blend."""
from shared import *
daily,intervals,staff,template=load_data()
feats={p:make_features(daily[p],p,staff) for p in PORTFOLIOS}
cv_prof,abd_prof,cct_prof,prof_cct_avg=build_profiles(intervals,daily)
aug=pd.date_range('2025-08-01','2025-08-31')
bias={'A':1.18,'B':1.20,'C':1.11,'D':1.18}
preds={}
for p in PORTFOLIOS:
    ft=feats[p];cols=feat_cols(ft);ok=ft[cols].notna().all(axis=1);cl=ft[ok].copy()
    trn=cl['Date']<'2025-07-01';val=(cl['Date']>='2025-07-01')&(cl['Date']<'2025-08-01')
    Xtr,Xv=cl.loc[trn,cols].values,cl.loc[val,cols].values
    preds[p]={};d=daily[p];a24=d[(d['Date'].dt.month==8)&(d['Date'].dt.year==2024)]
    for met in ['Call Volume','CCT']:
        ytr,yv=cl.loc[trn,f'tgt_{met}'].values,cl.loc[val,f'tgt_{met}'].values
        q=0.52 if met=='CCT' else 0.55
        gb=HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=500,max_depth=6,learning_rate=0.05,min_samples_leaf=10,random_state=42)
        gb.fit(Xtr,ytr);rd=Ridge(alpha=1.0);rd.fit(np.nan_to_num(Xtr,0),ytr)
        vp=0.6*gb.predict(Xv)+0.4*rd.predict(np.nan_to_num(Xv,0))
        print(f'  {p} {met} val MAE: {mean_absolute_error(yv,vp):.2f}')
        amsk=(ft['Date']>='2025-08-01')&(ft['Date']<='2025-08-31')
        Xa=ft.loc[amsk,cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        ml=0.6*gb.predict(Xa.values)+0.4*rd.predict(np.nan_to_num(Xa.values,0))
        # CHANGE: 100% ML, no baseline
        preds[p][met]=ml[:31]
    recent=d[(d['Date']>='2025-06-01')&(d['Date']<'2025-08-01')];abd=np.zeros(31)
    for i,dt in enumerate(aug):
        dw=dt.dayofweek;r=recent[recent['Date'].dt.dayofweek==dw]['Abandon Rate'];a=a24[a24['Date'].dt.dayofweek==dw]['Abandon Rate']
        if len(r)>0 and len(a)>0:abd[i]=0.6*r.mean()+0.4*a.mean()
        elif len(r)>0:abd[i]=r.mean()
        else:abd[i]=d['Abandon Rate'].tail(60).mean()
    abd*=1.1;abd=np.clip(abd,0.002,0.25);preds[p]['Abandon Rate']=abd
res={p:{'cv':[],'abd':[],'ar':[],'cct':[]} for p in PORTFOLIOS}
for p in PORTFOLIOS:
    dcv,dcct,dar=preds[p]['Call Volume'],preds[p]['CCT'],preds[p]['Abandon Rate'];dabd=dcv*dar
    for i,dt in enumerate(aug):
        dw=dt.dayofweek;cv=dcv[i]*cv_prof[p][dw];ab=dabd[i]*abd_prof[p][dw]
        sc=dcct[i]/prof_cct_avg[p] if prof_cct_avg[p]>0 else 1.0;cc=cct_prof[p][dw]*sc;ar=np.where(cv>0,ab/cv,0.0)
        res[p]['cv'].append(cv);res[p]['abd'].append(ab);res[p]['ar'].append(ar);res[p]['cct'].append(cc)
    for k in res[p]:res[p][k]=np.array(res[p][k])
for p in PORTFOLIOS:b=bias[p];res[p]['cv']*=b;res[p]['abd']*=b;res[p]['cct']*=1.05
for p in PORTFOLIOS:
    res[p]['cv']=np.clip(res[p]['cv'],0,None);res[p]['abd']=np.clip(res[p]['abd'],0,None);res[p]['cct']=np.clip(res[p]['cct'],0,None)
    bad=res[p]['abd']>res[p]['cv'];res[p]['abd'][bad]=res[p]['cv'][bad];cv,ab=res[p]['cv'],res[p]['abd']
    res[p]['ar']=np.clip(np.where(cv>0,ab/cv,0.0),0,1);res[p]['cv']=np.round(cv).astype(int);res[p]['abd']=np.round(res[p]['abd']).astype(int)
ratios,abd_rates=score_submission(res,daily)
print(f"\n>> NO BASELINE (ML only)")
for p in PORTFOLIOS:print(f"   {p}: ratio={ratios[p]:.3f}")
print(f"   Under: {sum(1 for r in ratios.values() if r<1.0)}/4")
