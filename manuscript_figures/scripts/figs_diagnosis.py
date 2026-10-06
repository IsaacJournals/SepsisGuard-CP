import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
T="../../results/tables/"
C={"AtoB":"#0072B2","BtoA":"#D55E00"}; G="#7F7F7F"; LAB={"AtoB":"A → B","BtoA":"B → A"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"axes.grid":True,"grid.color":"#e6e6e6","grid.linewidth":0.5,
    "axes.axisbelow":True,"legend.frameon":False,"savefig.dpi":600})
def save(f,n):
    f.savefig(f"../{n}.png",bbox_inches="tight"); plt.close(f)
r=lambda n: pd.read_csv(T+n+".csv")

# ---- Fig: calibration curves ----
cal=r("T49_Calibration_Intercept_Slope"); cur=r("T49b_Calibration_Curves")
from matplotlib.lines import Line2D
f,axs=plt.subplots(1,2,figsize=(7.2,3.9))
sty={"internal_test":(None,"black","-","o","Internal test"),"external_unchanged":(1,None,"-","o","External, unchanged"),"external_recalibrated_ALL":(2,None,"--","^","External, locally recalibrated")}
short={"internal_test":"Internal","external_unchanged":"External, unchanged","external_recalibrated_ALL":"External, recalibrated"}
for a,t in zip(axs,["AtoB","BtoA"]):
    top=0.16; txt=[]
    for s_,(_,col,ls,mk,lab) in sty.items():
        c=col or C[t]; z=cur[(cur.Direction==t)&(cur.Setting==s_)]
        sp=z[z.Type=="spline"]; dc=z[z.Type=="decile"]; q=cal[(cal.Direction==t)&(cal.Setting==s_)].iloc[0]
        a.plot(sp.Predicted,sp.Observed,color=c,ls=ls,lw=1.4); a.scatter(dc.Predicted,dc.Observed,color=c,s=10,zorder=3,marker=mk)
        txt.append(f"{short[s_]}: {q.CITL:.2f} / {q.Slope:.2f}")
    a.plot([0,top],[0,top],ls=":",c=G,lw=0.8)
    a.text(0.97,0.03,"Intercept / slope\n"+"\n".join(txt),transform=a.transAxes,ha="right",va="bottom",fontsize=6.3,
           bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#cccccc",lw=0.5))
    a.set(xlim=(0,top),ylim=(0,top),xlabel="Predicted hourly risk",ylabel="Observed hourly event rate",title=LAB[t]); a.set_aspect("equal")
h_=[Line2D([0],[0],color="black",lw=1.4,label="Internal test"),Line2D([0],[0],color="#555555",lw=1.4,label="External, unchanged"),
    Line2D([0],[0],color="#555555",lw=1.4,ls="--",label="External, locally recalibrated"),Line2D([0],[0],color=G,lw=0.8,ls=":",label="Perfect calibration")]
f.legend(handles=h_,loc="lower center",ncol=4,fontsize=6.8,bbox_to_anchor=(0.5,-0.02))
f.tight_layout(rect=(0,0.06,1,1)); save(f,"Fig3_Calibration_Curves")

# ---- Fig: diagnosis ----
s=r("T50_CaseMix_Measurement_Shift"); m=s[s.Domain=="measurement practice"].copy()
m["Var"]=m.Characteristic.str.split(":").str[0]; m["d"]=(m.EverRecorded_A-m.EverRecorded_B)
m["isdiv"]=(m.Diff_pp.abs()>5)|(m.d.abs()>0.10); m=m.sort_values("d")
h=r("T53b_Missingness_Hidden_Refit"); ref=r("T51_Measurement_Agnostic_Refit")
f=plt.figure(figsize=(7.2,5.0)); gs=f.add_gridspec(2,2,width_ratios=[1.05,1],hspace=0.45,wspace=0.35)
a=f.add_subplot(gs[:,0]); y=np.arange(len(m))
for i,(_,q) in enumerate(m.iterrows()):
    a.plot([q.EverRecorded_A,q.EverRecorded_B],[i,i],color="#333333" if q.isdiv else "#bbbbbb",lw=0.8)
a.scatter(m.EverRecorded_A,y,s=12,color="#009E73",label="Hospital A",zorder=3); a.scatter(m.EverRecorded_B,y,s=12,color="#CC79A7",marker="s",label="Hospital B",zorder=3)
a.set_yticks(y); a.set_yticklabels([v+(" *" if d else "") for v,d in zip(m.Var,m.isdiv)],fontsize=6)
a.set(xlabel="Share of ICU stays with ≥1 recorded value",xlim=(-0.02,1.02)); a.legend(fontsize=6.5,loc="lower center",bbox_to_anchor=(0.5,1.03),ncol=2); a.set_title("a  Recording practice",loc="left",fontsize=8,pad=18)
vs=[("Original model","ref"),("Recording hidden","h0"),("Recording hidden,\n21 variables removed","h1")]
def row(t,k):
    if k=="ref": q=ref[(ref.Direction==t)&ref.Variant.str.startswith("primary")].iloc[0]; return q.External_Slope,q.Slope_L,q.Slope_U,q.Patient_SepticCoverage,q.Patient_NonSepticCoverage
    q=h[(h.Direction==t)&(h.Variant==("missingness hidden (random-draw filling)" if k=="h0" else "missingness hidden, divergent variables removed"))].iloc[0]
    return q.External_Slope,q.Slope_L,q.Slope_U,q.Patient_SepticCoverage,q.Patient_NonSepticCoverage
b=f.add_subplot(gs[0,1]); c=f.add_subplot(gs[1,1])
for j,t in enumerate(["AtoB","BtoA"]):
    for i,(lab,k) in enumerate(vs):
        sl,lo,hi,ps,pn=row(t,k); x=i+(j-0.5)*0.25
        b.errorbar(x,sl,yerr=[[sl-lo],[hi-sl]],fmt="o",ms=4,color=C[t],capsize=2,lw=0.8,label=LAB[t] if i==0 else None)
        c.scatter(x,ps,color=C[t],marker="o",s=18,label=f"{LAB[t]} septic" if i==0 else None)
        c.scatter(x,pn,facecolor="white",edgecolor=C[t],marker="s",s=18,label=f"{LAB[t]} non-septic" if i==0 else None)
for ax_ in (b,c): ax_.set_xticks(range(3)); ax_.set_xticklabels([v[0] for v in vs],fontsize=6.3)
b.axhline(1,ls=":",c=G,lw=0.8); b.set(ylabel="External calibration slope",ylim=(0.6,1.05)); b.legend(fontsize=6.3,loc="lower right"); b.set_title("b  Calibration slope (95% CI)",loc="left",fontsize=8)
c.axhline(0.90,ls="--",c=G,lw=0.8); c.set(ylabel="Patient-level coverage",ylim=(0.5,1.02)); c.legend(fontsize=5.8,loc="lower right",ncol=2); c.set_title("c  Class-conditional coverage",loc="left",fontsize=8)
save(f,"Fig6_Diagnosis")

# ---- Supplementary: decision curves ----
for name,tab,xmax,lab in [("FigS5_Decision_Curves_Hourly","T52_Decision_Curve_Transfer",0.10,"Threshold probability (hourly)"),("FigS6_Decision_Curves_Patient","T56_Patient_Level_Decision_Curve",0.40,"Threshold probability (per ICU stay)")]:
    d=r(tab); f,axs=plt.subplots(1,2,figsize=(7.2,3.0))
    for a,t in zip(axs,["AtoB","BtoA"]):
        z=d[(d.Direction==t)&(d.ThresholdProb<=xmax)]; e=z[z.Setting=="external_unchanged"]
        a.plot(e.ThresholdProb,e.TreatAll,color=G,lw=0.9,label="Treat all"); a.axhline(0,color="black",lw=0.7,label="Treat none")
        a.plot(e.ThresholdProb,e.NetBenefit,color=C[t],lw=1.4,label="External, unchanged")
        u=z[z.Setting=="external_recalibrated_ALL"]; a.plot(u.ThresholdProb,u.NetBenefit,color=C[t],ls="--",lw=1.4,label="External, locally recalibrated")
        ymax=max(z.NetBenefit.max()*1.15,1e-4); a.set(xlim=(0,xmax),ylim=(-0.25*ymax,ymax),xlabel=lab,ylabel="Net benefit",title=LAB[t]); a.legend(fontsize=6.3)
    f.tight_layout(); save(f,name)
# ---- Supplementary: domain classifier families ----
dd=r("T50b_Domain_Classifier"); fam=dd[dd.Part.str.startswith("gain share")].sort_values("Value")
nm={"window_12_24h":"12/24-h window statistics","last_observed":"Last observed values","window_6h":"6-h window statistics","static_context":"Static and context","delta":"Changes","current":"Current values","missingness":"Missingness indicators","clinical_composite":"Clinical composites"}
f,a=plt.subplots(figsize=(5.5,2.8)); a.barh([nm.get(i,i) for i in fam.Item],fam.Value,color="#0072B2")
a.set(xlabel="Share of gain in a classifier separating hospital A from B"); f.tight_layout(); save(f,"FigS7_Domain_Classifier")
print("ok")
