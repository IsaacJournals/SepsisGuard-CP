import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans"})
f,ax=plt.subplots(figsize=(13/2.54*1.6,5/2.54*1.6)); ax.set_xlim(0,130); ax.set_ylim(0,50); ax.axis("off")
def box(x,y,w,h,t,fc,ec="#333333",fs=7.2,b=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.5,rounding_size=1.5",fc=fc,ec=ec,lw=0.8)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,fontweight="bold" if b else "normal")
def arr(x1,y1,x2,y2,c="#333333"): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=9,lw=0.9,color=c))
box(2,30,26,16,"Hospital A\n20,336 ICU stays",fc="#e6f0f8",b=True); box(2,4,26,16,"Hospital B\n20,000 ICU stays",fc="#fbe9e0",b=True)
box(36,17,26,16,"Locked protocol\nPlatt calibration\nclass-conditional\nconformal sets\nburden-matched alarm",fc="#f3f3f3",fs=6.6)
arr(28,38,36,28); arr(28,12,36,22)
box(70,30,27,16,"A → B unchanged:\nonly 53% of septic\npatients fully covered;\ndetection 0.70 → 0.46",fc="#e6f0f8",fs=6.6)
box(70,4,27,16,"B → A unchanged:\nfalse alarms\n0.18 → 0.40 per\npatient-day",fc="#fbe9e0",fs=6.6)
arr(62,28,70,38); arr(62,22,70,12)
box(104,17,24,16,"Local recalibration\nrestores coverage\nand alarm budget;\nreliable with\n~800 local patients",fc="#e8f4ec",fs=6.6,b=False)
arr(97,38,104,28); arr(97,12,104,22)
ax.text(116,11,"+28 to +41 points\nvs partial NEWS2\nat equal burden",ha="center",va="center",fontsize=6.4,style="italic")
f.savefig("../Graphical_Abstract.png",dpi=600,bbox_inches="tight"); print("ok")
