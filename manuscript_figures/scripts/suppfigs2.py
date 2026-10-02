import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#e6e6e6","axes.axisbelow":True,"legend.frameon":False,"savefig.dpi":600})
d=pd.read_pickle("../../results/predictions/predictions_test_rows.pkl")
M=[("SepsisGuard_CP","Calibrated ensemble","#000000","-"),("LightGBM","LightGBM","#0072B2","-"),("XGBoost","XGBoost","#56B4E9","--"),("RandomForest","Random forest","#009E73","-."),
   ("ExtraTrees","Extremely randomized trees","#E69F00","--"),("Logistic","Logistic regression","#D55E00",":"),("CompactGRU","GRU","#CC79A7","-.")]
f,axs=plt.subplots(2,2,figsize=(7.2,6.0))
for j,(sp,tt) in enumerate([("A_test","Hospital A, internal test"),("B_test","Hospital B, external test")]):
    s=d[d.split==sp]; y=s.y.values
    for m,l,c,ls in M:
        p=s[m].values; ok=~np.isnan(p)
        fpr,tpr,_=roc_curve(y[ok],p[ok]); pr,rc,_=precision_recall_curve(y[ok],p[ok])
        axs[0,j].plot(fpr,tpr,ls=ls,color=c,lw=1.1,label=f"{l} ({roc_auc_score(y[ok],p[ok]):.3f})")
        axs[1,j].plot(rc,pr,ls=ls,color=c,lw=1.1,label=f"{l} ({average_precision_score(y[ok],p[ok]):.3f})")
    axs[0,j].plot([0,1],[0,1],c="#999999",lw=0.6); axs[0,j].set(xlabel="False-positive rate",ylabel="True-positive rate",title=tt+": ROC (AUROC)")
    axs[1,j].axhline(y.mean(),c="#999999",lw=0.6); axs[1,j].set(xlabel="Recall",ylabel="Precision",title=tt+": precision–recall (AUPRC)",ylim=(0,0.5))
    for a in axs[:,j]: a.legend(fontsize=6.2,loc="lower right" if a in axs[0] else "upper right"); a.title.set_fontsize(8)
f.tight_layout(); f.savefig("../FigS3_ROC_PR_Development.png",bbox_inches="tight"); plt.close(f)
# reliability (decile bins)
def rel(y,p,q=10):
    b=pd.qcut(p,q,labels=False,duplicates="drop"); g=pd.DataFrame({"y":y,"p":p,"b":b}).groupby("b").mean(); return g.p.values,g.y.values
f,axs=plt.subplots(1,2,figsize=(7.2,3.0))
a=d[d.split=="A_test"]; b=d[d.split=="B_test"]
for ax,(lst,tt) in zip(axs,[([(a,"Ensemble_Raw","Raw ensemble","#D55E00"),(a,"SepsisGuard_CP","Platt-recalibrated ensemble","#0072B2")],"Hospital A, internal test"),
                             ([(b,"SepsisGuard_CP","Unchanged (hospital A calibration)","#D55E00"),(b,"SG_B_platt_recal","Platt refitted on 4,000 local patients","#0072B2")],"Hospital B, external test")]):
    mx=0
    for s,m,l,c in lst:
        ok=~s[m].isna(); px,oy=rel(s.y.values[ok],s[m].values[ok]); ax.plot(px,oy,"o-",ms=3,color=c,lw=1.1,label=l); mx=max(mx,px.max(),oy.max())
    ax.plot([0,mx],[0,mx],ls=":",c="#777777",lw=0.8); ax.set(xlabel="Mean predicted risk (decile bins)",ylabel="Observed event rate",title=tt); ax.title.set_fontsize(8); ax.legend(fontsize=6.5)
f.tight_layout(); f.savefig("../FigS4_Reliability_Development.png",bbox_inches="tight"); plt.close(f)
print("ok", d.SG_B_platt_recal.notna().sum())
