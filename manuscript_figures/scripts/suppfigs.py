import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
T="../../results/tables/"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#e6e6e6","axes.axisbelow":True,"legend.frameon":False,"savefig.dpi":600})
c=pd.read_csv(T+"T10_Conformal_By_Hours_From_Onset.csv"); c=c[c.HoursFromOnset!="> 3 h"]
bins=["<= -12 h","(-12,-6] h","(-6,0] h","(0,3] h","non-septic patients"]; bl=["≤ −12 h","−12 to −6 h","−6 to 0 h","0 to +3 h","Non-septic\npatients"]
M=[("mondrian","Mondrian (hour level)","#0072B2"),("patient_mondrian","Class-conditional patient level","#D55E00"),("marginal","Marginal","#7F7F7F")]
f,axs=plt.subplots(1,2,figsize=(7.2,2.8),sharey=True)
for a,(sp,tt) in zip(axs,[("A_test","Hospital A, internal test"),("B_test_unchanged","Hospital B, unchanged transfer")]):
    x=np.arange(len(bins)); w=0.26
    for k,(m,l,col) in enumerate(M):
        s=c[(c.Split==sp)&(c.Method==m)].set_index("HoursFromOnset").loc[bins]
        a.bar(x+(k-1)*w,s.Coverage,w,color=col,label=l)
    a.axhline(0.9,ls="--",c="#444444",lw=0.8); a.set_xticks(x); a.set_xticklabels(bl,fontsize=7); a.set_title(tt,fontsize=8); a.set_ylim(0,1.05); a.grid(axis="x",visible=False)
axs[0].set_ylabel("Hour-level coverage"); axs[0].set_xlabel("Hours relative to t_sepsis (septic patients)",fontsize=7); axs[1].set_xlabel("Hours relative to t_sepsis (septic patients)",fontsize=7)
h,l=axs[0].get_legend_handles_labels(); f.legend(h,l,loc="lower center",ncol=3,fontsize=7,bbox_to_anchor=(0.5,-0.06))
f.tight_layout(rect=(0,0.07,1,1)); f.savefig("../FigS1_Coverage_By_Time.png",bbox_inches="tight"); plt.close(f)
pm=pd.read_csv(T+"T20_Family_Importance_Ensemble_Permutation.csv")
nm={"static_context":"Static and context","window_12_24h":"12/24-h window","last_observed":"Last observed","window_6h":"6-h window","current":"Current value","clinical_composite":"Clinical composites","delta":"Change","missingness":"Missingness"}
sh=pd.read_csv(T+"T19b_SHAP_By_Clinical_Variable.csv").head(15)
vn={"ICULOS":"Hours since ICU admission","HospAdmTime":"Time from hospital admission","Unit1":"Unit indicator (MICU)","Unit2":"Unit indicator (SICU)","Temp":"Temperature","Resp":"Respiratory rate","Hct":"Hematocrit","BUN":"Urea nitrogen","FiO2":"FiO2","PTT":"PTT","SBP":"Systolic BP","MAP":"Mean arterial pressure","WBC":"White cell count","HR":"Heart rate","Creatinine":"Creatinine","Calcium":"Calcium","Lactate":"Lactate","Age":"Age","Platelets":"Platelets","Glucose":"Glucose","O2Sat":"Oxygen saturation","ShockIndex":"Shock index","Magnesium":"Magnesium","Potassium":"Potassium","Hgb":"Hemoglobin","Chloride":"Chloride","Phosphate":"Phosphate","DBP":"Diastolic BP","pH":"pH","BaseExcess":"Base excess","HCO3":"Bicarbonate","PaCO2":"PaCO2","SaO2":"SaO2","AST":"AST","Alkalinephos":"Alkaline phosphatase","Bilirubin_total":"Total bilirubin","Bilirubin_direct":"Direct bilirubin","TroponinI":"Troponin I","Fibrinogen":"Fibrinogen","EtCO2":"EtCO2","Gender":"Sex","BUN_Creatinine":"Urea/creatinine ratio","MeasCount24":"Measurements in 24 h","qSOFA_partial":"Partial qSOFA","SIRS_partial":"Partial SIRS"}
f,axs=plt.subplots(1,2,figsize=(7.2,3.2))
pm=pm.sort_values("AUPRC_drop_mean")
axs[0].barh([nm[x] for x in pm.Family],pm.AUPRC_drop_mean,xerr=pm.AUPRC_drop_sd,color="#0072B2",height=0.6)
axs[0].set_xlabel("AUPRC decrease when family permuted"); axs[0].set_title("Ensemble: grouped permutation importance",fontsize=8); axs[0].grid(axis="y",visible=False)
sh=sh.iloc[::-1]
axs[1].barh([vn.get(v,v) for v in sh.Variable],sh.MeanAbsSHAP,color="#D55E00",height=0.6)
axs[1].set_xlabel("Mean |SHAP| summed over a variable's features"); axs[1].set_title("LightGBM member: TreeSHAP by clinical variable",fontsize=8); axs[1].grid(axis="y",visible=False)
f.tight_layout(); f.savefig("../FigS2_Explainability.png",bbox_inches="tight"); plt.close(f)
print(set(pd.read_csv(T+"T19b_SHAP_By_Clinical_Variable.csv").head(15).Variable)-set(vn))
