import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
T="../../results/tables/"
C={"AtoB":"#0072B2","BtoA":"#D55E00"}; G="#7F7F7F"; LAB={"AtoB":"A → B","BtoA":"B → A"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"axes.grid":True,"grid.color":"#e6e6e6","grid.linewidth":0.5,
    "axes.axisbelow":True,"legend.frameon":False,"savefig.dpi":600})
def save(f,n):
    f.savefig(f"../{n}.png",bbox_inches="tight"); f.savefig(f"../{n}.pdf",bbox_inches="tight"); plt.close(f)

# ---------- Fig 1: study design ----------
f,ax=plt.subplots(figsize=(7.2,4.2)); ax.set_xlim(0,100); ax.set_ylim(-1.5,50); ax.axis("off"); ax.grid(False)
def box(x,y,w,h,t,fc="#ffffff",ec="#333333",fs=7,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc=fc,ec=ec,lw=0.7))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal",wrap=True)
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=8,lw=0.7,color="#333333"))
box(1,39,30,10,"Development hospital\n(A or B)",fc="#eef4fa",bold=True)
box(1,27,30,10,"TRAIN 60% | VA_FIT 10%\nVA_CONF 10% | TEST 20%\n(patient-level, stratified)",fs=6.5)
box(1,13,30,11,"Past-only features (351)\nCandidate models → locked\nprimary-model rule (VA_FIT)",fs=6.5)
box(1,1,30,9,"Platt (VA_FIT) · Mondrian and\npatient-level conformal (VA_CONF)\nBurden-matched threshold",fs=6.5)
arr(16,39,16,37.6); arr(16,27,16,24.6); arr(16,13,16,10.6)
box(36,39,28,10,"Internal test\n(development hospital)",fc="#f3f3f3")
box(36,24,28,12,"External hospital\nLOCAL pool 20%\nEXTERNAL TEST 80%",fc="#fbefe8",bold=False,fs=6.5)
arr(31.5,5,36,41); arr(31.5,5,36,29)
box(69,39,30,10,"Arm 1: unchanged transfer\n(model, calibration,\nquantiles, threshold)",fs=6.5)
box(69,27,30,10,"Arm 2: local recalibration\non n patients (n = 25 … all;\n30 resamples per n)",fs=6.5)
box(69,14,30,10,"Arm 3: retraining +\nrecalibration\n(n = 100, 400, 1,600, all)",fs=6.5)
box(69,0,30,11,"Endpoints: AUROC, AUPRC,\ncalibration intercept and slope,\nclass-conditional coverage, detection\n≤48 h at ≤0.20 false alarms/patient-day",fs=6.0)
for y in (44,32,19): arr(64.5,30,69,y)
ax.text(50,16,"Run in both directions\nA → B and B → A\nunder one protocol\n(SHA-256 locked\nbefore analysis)",ha="center",va="center",fontsize=6.8,style="italic")
box(36,1,28,9,"Post hoc: causes of miscalibration\n(prevalence, case mix, recording\npractice); net benefit",fc="#f6f6f6",fs=6.0)
save(f,"Fig1_Study_Design")

# ---------- Fig 2: discrimination forest ----------
p=pd.read_csv(T+"T41_v9_Performance_Both_Directions.csv")
order=["LightGBM_tuned","Ensemble","LightGBM","LightGBM_noTime","NEWS2_partial"]
nm={"LightGBM_tuned":"Tuned LightGBM (primary)","Ensemble":"Weighted ensemble","LightGBM":"LightGBM (default)","LightGBM_noTime":"LightGBM without time features","NEWS2_partial":"Partial NEWS2"}
f,axs=plt.subplots(2,2,figsize=(7.2,4.2),sharey=True)
for j,met in enumerate(["AUROC","AUPRC"]):
    for i,sp in enumerate(["internal_test","external_unchanged"]):
        a=axs[j,i]
        for k,d in enumerate(["AtoB","BtoA"]):
            s=p[(p.Split==sp)&(p.Direction==d)].set_index("Model").loc[order]
            y=np.arange(len(order))[::-1]+(0.15 if d=="AtoB" else -0.15)
            a.errorbar(s[met],y,xerr=[s[met]-s[met+"_L"],s[met+"_U"]-s[met]],fmt="o",ms=4,color=C[d],lw=1,capsize=1.5,label=LAB[d])
        a.set_yticks(np.arange(len(order))[::-1]); a.set_yticklabels([nm[o] for o in order])
        a.set_title(("Internal test" if sp=="internal_test" else "External test (unchanged)")+f" — {met}",fontsize=8)
        a.grid(axis="y",visible=False)
h,l=axs[0,0].get_legend_handles_labels(); f.legend(h,l,loc="upper center",ncol=2,fontsize=7.5,bbox_to_anchor=(0.6,1.03))
f.tight_layout(rect=(0,0,1,0.96)); save(f,"Fig2_Discrimination_Both_Directions")

# ---------- Fig 3: class-conditional coverage ----------
t=pd.read_csv(T+"T43_v9_Transfer_Conformal_Calibration.csv")
f,axs=plt.subplots(1,2,figsize=(7.2,3.0))
splits=["internal_test","external_unchanged","external_recalibrated_ALL"]; sl=["Internal\ntest","External,\nunchanged","External,\nrecalibrated"]
for a,d in zip(axs,["AtoB","BtoA"]):
    s=t[(t.Direction==d)&(t.Method=="patient_mondrian")].set_index("Split").loc[splits]
    x=np.arange(3); w=0.36
    b1=a.bar(x-w/2-0.01,s.PatientCov_SepticHours,w,color=C[d],label="Septic patients: all septic hours covered")
    b2=a.bar(x+w/2+0.01,s.PatientCov_NonSepticHours,w,color="white",edgecolor=C[d],hatch="////",lw=0.8,label="Non-septic patients: all hours covered")
    for bars in (b1,b2):
        for r in bars: a.text(r.get_x()+r.get_width()/2,r.get_height()+0.01,f"{r.get_height():.2f}",ha="center",va="bottom",fontsize=6.5)
    a.axhline(0.90,ls="--",c=G,lw=0.8); a.text(-0.62,0.86,"nominal\n0.90",fontsize=6,color=G,ha="left",va="top")
    a.set_xticks(x); a.set_xticklabels(sl); a.set_ylim(0,1.08); a.set_xlim(-0.65,2.6); a.set_title(LAB[d],fontsize=8.5,color=C[d],fontweight="bold")
    a.grid(axis="x",visible=False)
axs[0].set_ylabel("Patient-level class-conditional coverage")
h,l=axs[0].get_legend_handles_labels(); f.legend(h,l,loc="lower center",fontsize=6.8,ncol=2,bbox_to_anchor=(0.5,-0.02))
f.tight_layout(rect=(0,0.07,1,1)); save(f,"Fig4_Class_Conditional_Coverage")

# ---------- Fig 4: clinical endpoint ----------
cl=pd.read_csv(T+"T42_v9_Clinical_Endpoint.csv"); eq=pd.read_csv(T+"T47_v91_H4_Equal_Burden_vs_NEWS2.csv")
f,axs=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True)
for a,d in zip(axs,["AtoB","BtoA"]):
    c=cl[cl.Direction==d]
    def pt(model,setting): r=c[(c.Model.str.contains(model))&(c.Setting==setting)].iloc[0]; return r.FalseEpisodesPerPatientDay,r.Detected48h
    pts=[("Primary, internal","LightGBM","internal test, VA_FIT burden threshold","o",C[d],"full"),
         ("Primary, external unchanged","LightGBM","external, unchanged threshold","s",C[d],"none"),
         ("Primary, external recalibrated","LightGBM","external, ALL local patients recalibrated + threshold","D",C[d],"full"),
         ("Partial NEWS2, internal","NEWS2","internal test, VA_FIT burden threshold","o",G,"full"),
         ("Partial NEWS2, external","NEWS2","external, unchanged threshold","s",G,"none")]
    for lab,m,s_,mk,col,fill in pts:
        x,y=pt(m,s_); a.plot(x,y,mk,ms=6,color=col,mfc=col if fill=="full" else "white",mew=1.1,label=lab)
    e=eq[(eq.Direction==d)&(eq.Split=="internal_test")].iloc[0]
    a.plot(e.Primary_FalseEpisodesPerPatientDay,e.Primary_Detected48h,"^",ms=6,color=C[d],mfc="white",mew=1.1,label="Primary, internal, NEWS2-matched burden")
    a.annotate("",xy=(e.Primary_FalseEpisodesPerPatientDay,e.Primary_Detected48h),xytext=(e.NEWS2_FalseEpisodesPerPatientDay,e.NEWS2_Detected48h),
               arrowprops=dict(arrowstyle="-",ls=":",color="#444444",lw=0.8))
    a.text((e.Primary_FalseEpisodesPerPatientDay+e.NEWS2_FalseEpisodesPerPatientDay)/2+0.008,(e.Primary_Detected48h+e.NEWS2_Detected48h)/2,
           f"+{100*e.Diff:.0f} points\n({100*e.Diff_L:.0f}–{100*e.Diff_U:.0f})",fontsize=6.3,va="center")
    a.axvline(0.20,ls="--",c=G,lw=0.8); a.text(0.203,0.05,"budget 0.20",fontsize=6,color=G,rotation=90,va="bottom")
    a.set_xlim(0,0.45); a.set_xticks([0,0.1,0.2,0.3,0.4]); a.set_ylim(0,0.8); a.set_xlabel("False-alarm episodes per non-septic patient-day")
    a.set_title(LAB[d],fontsize=8.5,color=C[d],fontweight="bold")
axs[0].set_ylabel("Septic patients alarmed within 48 h")
h,l=axs[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
gen=[Line2D([],[],marker="o",ls="",color="k",label="Internal test, protocol threshold"),
     Line2D([],[],marker="^",ls="",color="k",mfc="white",label="Internal test, NEWS2-matched burden"),
     Line2D([],[],marker="s",ls="",color="k",mfc="white",label="External test, unchanged threshold"),
     Line2D([],[],marker="D",ls="",color="k",label="External test, local recalibration + threshold"),
     Line2D([],[],marker="o",ls="",color=G,label="Partial NEWS2 (grey)")]
f.legend(handles=gen,loc="lower center",ncol=3,fontsize=6.5,bbox_to_anchor=(0.5,-0.1))
f.tight_layout(rect=(0,0.06,1,1)); save(f,"Fig5_Clinical_Endpoint")

# ---------- Fig 5: budget curve (percentile bands, degenerate shaded) ----------
raw=pd.read_csv(T+"T44b_v9_Recalibration_Budget_Raw.csv"); rob=pd.read_csv(T+"T48_v91_Budget_Robustness_Degeneracy.csv")
raw["Deg"]=raw.LocalSeptic<9
f,axs=plt.subplots(1,3,figsize=(7.2,2.6))
for d in ["AtoB","BtoA"]:
    r=raw[raw.Direction==d]; ue=rob[rob.Direction==d].Unchanged_ECE.iloc[0]
    for a,k,excl in zip(axs[:2],["PatientSepticCoverage","ECE"],[True,False]):
        g=(r[~r.Deg] if excl else r).groupby("LocalPatients")[k]
        n_ok=(r[~r.Deg] if excl else r).groupby("LocalPatients").size(); keep=n_ok[n_ok>=10].index
        med,lo,hi=g.median().loc[keep],g.quantile(.025).loc[keep],g.quantile(.975).loc[keep]
        a.plot(med.index,med.values,"o-",ms=3,lw=1.2,color=C[d],label=LAB[d]); a.fill_between(med.index,lo.values,hi.values,color=C[d],alpha=0.15,lw=0)
    axs[1].axhline(ue,ls=":",c=C[d],lw=0.9)
    s=rob[rob.Direction==d]; axs[2].plot(s.LocalPatients,100*s.Share_meeting_H3_criteria,"o-",ms=3,lw=1.2,color=C[d],label=LAB[d])
for a in axs: a.set_xscale("log"); a.set_xlabel("Labeled local patients")
axs[0].axhline(0.90,ls="--",c=G,lw=0.8); axs[0].axhline(0.85,ls=":",c=G,lw=0.8); axs[0].set_ylabel("Septic patients fully covered"); axs[0].set_ylim(0.6,1.01)
axs[0].set_title("Patient-level septic coverage",fontsize=8); axs[1].set_title("External ECE (dotted: unchanged)",fontsize=8)
axs[1].set_ylabel("ECE")
axs[2].axhline(80,ls="--",c=G,lw=0.8); axs[2].set_ylabel("Resamples meeting H3 criteria (%)"); axs[2].set_title("Share of simulated sites",fontsize=8); axs[2].set_ylim(0,105)
axs[2].legend(fontsize=7,loc="lower right")
f.tight_layout(); save(f,"Fig7_Recalibration_Budget")

# ---------- Fig 6: retrain vs recalibrate ----------
rt=pd.read_csv(T+"T45_v9_Retrain_vs_Recalibrate.csv")
f,axs=plt.subplots(1,2,figsize=(7.2,2.7))
for d in ["AtoB","BtoA"]:
    r=rt[(rt.Direction==d)&(rt.Arm=="retrain + recalibrate")]; ref=rt[(rt.Direction==d)&(rt.Arm!="retrain + recalibrate")].iloc[0]
    for a,k in zip(axs,["AUROC_ext","Detected48h"]):
        a.scatter(r.LocalPatients*(1.04 if d=="AtoB" else 0.96),r[k],s=12,color=C[d],alpha=0.55,lw=0)
        m=r.groupby("LocalPatients")[k].mean(); a.plot(m.index,m.values,"-",lw=1.2,color=C[d],label=f"{LAB[d]} retrain + recalibrate")
        a.axhline(ref[k],ls=":",lw=1,color=C[d],label=f"{LAB[d]} recalibrate only (all local)")
for a,k in zip(axs,["External AUROC","Septic patients alarmed within 48 h"]): a.set_xscale("log"); a.set_xlabel("Labeled local patients"); a.set_ylabel(k)
axs[0].legend(fontsize=6.3,loc="upper left")
f.tight_layout(); save(f,"Fig8_Retrain_vs_Recalibrate")
print("ok")
