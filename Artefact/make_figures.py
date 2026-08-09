#!/usr/bin/env python3
"""
make_figures.py - regenerates the dissertation figures from the frozen CSVs.

USAGE
    python make_figures.py --repo /path/to/project --out figures

Reads your frozen result files where it can find them. Where a value is
already fixed in the dissertation text it falls back to that constant and
says so, so you always know whether a figure came from your data or from a
transcribed number.

Two figures are CSV-only and will SKIP rather than guess, because I do not
have their values anywhere else:
    fig_mechanism        needs coverage_matrix.csv mechanism columns
    fig_detection_f1     needs results/step10/detection_metrics.csv

Figure 1 (experimental design) is not produced here. It is drawn in TikZ
inside the .tex so it stays vector and always readable.

WHY THE FIGURES ARE THIS SIZE
The old figures were drawn 3700-5600 px wide and then squeezed into ~480 pt
of text width, a scale factor near 0.08, so 10 pt source text printed at
about 1 pt. Everything here is drawn at close to final printed size.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    import pandas as pd
except ImportError:
    pd = None

W = 6.7
TEAL, BLUE, GREY, RED, DARK, AMBER = "#2E8B76", "#3E7CB1", "#D9D9D9", "#C0392B", "#1A1A1A", "#E8A33D"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 1.0, "lines.linewidth": 2.0,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

ATTACKS = ["TokenBreak", "AdvTok", "Invisible", "Homoglyph", "Compatibility", "Reorder"]
REAL = ["TokenBreak", "AdvTok", "Homoglyph", "Compatibility", "Reorder"]
DEFENCES = ["Tokenizer translation", "Canonical reject", "Canonical replace", "Unicode sanitiser",
            "NFKC + confusables", "Global CPT", "Window CPT"]
CATS = ["Code", "URLs", "Emoji", "Misspelling", "Non-English", "Mixed-script"]
C, P, N, H, B = "COMPLETE", "PARTIAL", "NO EFFECT", "HARMFUL", "NO BASE"

# ---- values transcribed from the dissertation / claims ledger (all verified there)
F_UNDEF = [("TokenBreak",150),("AdvTok",150),("Compatibility",142),("Reorder",138),
           ("Homoglyph",134),("Invisible",0)]
F_COVER = {
 "Tokenizer translation":[(P,46.5),(C,0.0),(H,13.4),(P,74.0),(C,0.0),(P,74.8)],
 "Canonical reject":     [(N,96.2),(C,0.0),(B,0.0),(N,85.9),(N,91.0),(N,88.5)],
 "Canonical replace":    [(N,96.2),(C,0.0),(B,0.0),(N,85.9),(N,91.0),(N,88.5)],
 "Unicode sanitiser":    [(N,96.2),(C,0.0),(B,0.0),(N,85.9),(N,91.0),(C,0.0)],
 "NFKC + confusables":   [(N,96.2),(C,0.0),(B,0.0),(C,0.0),(C,0.0),(N,88.5)],
 "Global CPT":           [(P,84.6),(P,2.7),(B,0.0),(P,69.8),(N,89.9),(N,88.6)],
 "Window CPT":           [(N,96.7),(P,84.1),(B,0.0),(N,86.1),(N,92.1),(N,89.4)]}
F_FOREST = [("Tokenizer translation",48.8,40.2,57.5,None),("Global CPT",11.4,6.7,16.8,None),
            ("Canonical reject",0.0,None,None,96.2),("Canonical replace",0.0,None,None,96.2),
            ("Unicode sanitiser",0.0,None,None,96.2),("NFKC + confusables",0.0,None,None,96.2),
            ("Window CPT",0.0,None,None,96.7)]
F_BLOCK = np.array([[0,0,3,1,4,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                    [0,0,0,0,0,0],[18,100,12,5,100,96],[49,7,0,1,80,80]],float)
F_CHANGE = np.array([[100,100,76,100,100,100],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                     [0,0,0,1,39,37],[0,0,0,0,0,0],[0,0,0,0,0,0]],float)
F_PRESERVE = {"Tokenizer translation":98.7,"Canonical reject":100.0,"Canonical replace":100.0,
              "Unicode sanitiser":100.0,"NFKC + confusables":100.0,"Global CPT":44.8,"Window CPT":63.8}
F_BINS = [("(0.5, 0.6]",10),("(0.6, 0.8]",18),("(0.8, 1.0]",128)]
F_MARGIN = {"TokenBreak":[10,17,123],"AdvTok":[10,17,123],"Invisible":[0,0,0],
            "Homoglyph":[10,16,108],"Compatibility":[10,17,115],"Reorder":[10,16,112]}
F_CALIB = dict(w5_at_floor=14.46, w10_at_floor=1.26)   # ledger: 5-token floor 1.0000; 10-token 1.1

REPO, OUT, FELLBACK = ".", "figures", []


def find(*rel):
    for r in rel:
        p = os.path.join(REPO, r)
        if os.path.exists(p):
            return p


def table(name, loader, fallback):
    if pd is not None:
        try:
            v = loader()
            if v is not None:
                print(f"  [csv  ] {name}"); return v
        except Exception as e:
            print(f"  [warn ] {name}: {e}")
    FELLBACK.append(name); print(f"  [const] {name}")
    return fallback


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))/d
    return max(0,(c-h))*100, min(1,(c+h))*100


def pct(x):
    x = float(x)
    return x*100 if x <= 1.0 else x


# --------------------------------------------------------------------- loaders
def load_cover():
    p = find("results/final_current/coverage_matrix.csv", "coverage_matrix.csv")
    if not p: return None
    df = pd.read_csv(p)
    sm = {"complete":C,"partial":P,"no_clear_effect":N,"harmful":H,"no_base_attack":B}
    out = {}
    for d in DEFENCES:
        key = d.split()[0].lower()
        row = []
        for a in ATTACKS:
            r = df[df["defence"].astype(str).str.lower().str.contains(key) &
                   df["attack"].astype(str).str.lower().str.contains(a.lower())]
            if r.empty: return None
            r = r.iloc[0]
            row.append((sm.get(str(r["coverage_status"]).strip().lower(), N),
                        pct(r["paired_defended_asr"])))
        out[d] = row
    return out


def load_margin():
    p = find("results/final_current/margin_stratified_asr.csv")
    if not p: return None
    df = pd.read_csv(p); out = {}
    for a in ATTACKS:
        s = df[df["attack"].astype(str).str.lower().str.contains(a.lower())]
        if len(s) != 3: return None
        out[a] = [int(v) for v in s.sort_values(s.columns[1])["successes"]]
    return out


def load_mechanism():
    """CSV-only. Returns {defence: {label: count}} or None."""
    p = find("data/defence_matrix.csv")
    if not p: return None
    df = pd.read_csv(p)
    if "attack_mechanism" not in df.columns or "defence" not in df.columns: return None
    out = {}
    for d in DEFENCES:
        key = d.split()[0].lower()
        s = df[df["defence"].astype(str).str.lower().str.contains(key)]
        if s.empty: return None
        out[d] = s["attack_mechanism"].value_counts().to_dict()
    return out


def load_detection():
    p = find("results/step10/detection_metrics.csv", "detection_metrics.csv")
    if not p: return None
    return pd.read_csv(p)


# --------------------------------------------------------------------- figures
def fig_undefended(u):
    fig, ax = plt.subplots(figsize=(W, 3.0))
    o = sorted(u, key=lambda r: -r[1])
    names = [x[0] for x in o][::-1]; cnt = [x[1] for x in o][::-1]
    vals = [100*c/156 for c in cnt]
    ax.barh(names, vals, color=BLUE, edgecolor=DARK, lw=1.2, height=.66)
    for i,(v,c) in enumerate(zip(vals,cnt)):
        ax.text(v+1.5, i, f"{v:.1f}%  ({c}/156)", va="center", fontsize=9.5)
    ax.set_xlim(0,126); ax.set_xticks(range(0,101,20))
    ax.set_xlabel("Attack success rate (%)")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x",ls=":",color="#BBB")
    fig.savefig(f"{OUT}/fig_undefended_asr.png"); plt.close(fig)


def fig_coverage(cov):
    fill={C:TEAL,P:BLUE,N:GREY,H:RED,B:"#F7F7F7"}; tc={C:"white",P:"white",N:DARK,H:"white",B:"#666"}
    fig, ax = plt.subplots(figsize=(W,4.4)); nr,nc=len(DEFENCES),len(ATTACKS)
    for r,d in enumerate(DEFENCES):
        for c in range(nc):
            st,asr = cov[d][c]
            ax.add_patch(plt.Rectangle((c,nr-1-r),1,1,fc=fill[st],ec="white",lw=2.2))
            lab = "NO BASE\nATTACK" if st==B else f"{st}\n{asr:.1f}%"
            ax.text(c+.5,nr-1-r+.5,lab,ha="center",va="center",fontsize=8,color=tc[st],
                    fontweight="bold" if st in (C,H) else "normal",linespacing=1.3)
    ax.set_xlim(0,nc); ax.set_ylim(0,nr)
    ax.set_xticks(np.arange(nc)+.5); ax.set_xticklabels(ATTACKS,fontsize=9.5)
    ax.set_yticks(np.arange(nr)+.5)
    ax.set_yticklabels([d.replace(" + ","+\n").replace("Tokenizer ","Tokenizer\n") for d in DEFENCES][::-1],fontsize=8.5)
    ax.set_xlabel("Attack"); ax.set_ylabel("Defence"); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    hs=[plt.Rectangle((0,0),1,1,fc=fill[k],ec="#999") for k in (C,P,N,H,B)]
    ax.legend(hs,["Complete","Partial","No clear effect","Harmful","No base attack"],
              loc="lower center",bbox_to_anchor=(.5,1.02),ncol=5,frameon=False,fontsize=9)
    fig.savefig(f"{OUT}/fig_coverage_matrix.png"); plt.close(fig)


def fig_tradeoff(cov, pres):
    """Headline figure: what each defence stops, against what it costs everyone else."""
    pts = {}
    for d in DEFENCES:
        x = round(float(np.mean([cov[d][ATTACKS.index(a)][1] for a in REAL])), 2)
        pts[d] = (x, pres[d])
    # merge defences that land on the same point so labels do not stack
    merged, seen = {}, {}
    for d, xy in pts.items():
        seen.setdefault(xy, []).append(d)
    for xy, ds in seen.items():
        if len(ds) > 1 and all(x.startswith("Canonical") for x in ds):
            merged["Canonical reject / replace"] = xy
        else:
            for d in ds: merged[d] = xy
    off = {"NFKC + confusables": (-14, 13), "Tokenizer translation": (18, -22),
           "Unicode sanitiser": (0, 32), "Canonical reject / replace": (0, 13),
           "Global CPT": (0, -22), "Window CPT": (0, -22)}
    fig, ax = plt.subplots(figsize=(W*.88, 4.0))
    ax.axhspan(30, 70, color=RED, alpha=.06)
    ax.text(103, 31.8, "heavy cost to ordinary users", ha="right", fontsize=8.5, color=RED)
    for d,(x,y) in merged.items():
        ax.scatter(x, y, s=150, color=BLUE, edgecolor=DARK, lw=1.4, zorder=3)
        ax.annotate(d, (x, y), textcoords="offset points",
                    xytext=off.get(d, (0, 12)), ha="center", fontsize=8.8, zorder=4)
    ax.set_xlabel("Attacks still getting through (%), averaged over the five real attacks")
    ax.set_ylabel("Awkward but legitimate text\nstill handled normally (%)")
    ax.set_xlim(28, 104); ax.set_ylim(30, 124)
    ax.set_yticks(range(30, 101, 10))
    ax.spines[["top","right"]].set_visible(False); ax.grid(ls=":", color="#CCC")
    fig.savefig(f"{OUT}/fig_security_utility_tradeoff.png"); plt.close(fig)


def fig_utility(blk, chg):
    reds=LinearSegmentedColormap.from_list("r",["#FFF5F0","#FCAE91","#FB6A4A","#CB181D","#67000D"])
    fig, axes = plt.subplots(1,2,figsize=(W,3.4))
    for ax,M,t in zip(axes,[blk,chg],["A. Legitimate text blocked (%)","B. Legitimate text rewritten (%)"]):
        ax.imshow(M,cmap=reds,vmin=0,vmax=100,aspect="auto")
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                v=M[r,c]
                ax.text(c,r,f"{v:.0f}",ha="center",va="center",fontsize=9,fontweight="bold",
                        color="white" if v>=55 else DARK)
        ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS,rotation=35,ha="right",fontsize=9)
        ax.set_title(t,fontsize=9.5,pad=6)
        ax.set_xticks(np.arange(-.5,len(CATS),1),minor=True)
        ax.set_yticks(np.arange(-.5,M.shape[0],1),minor=True)
        ax.grid(which="minor",color="white",lw=2.0)
        ax.tick_params(which="minor",length=0); ax.tick_params(length=0)
    axes[0].set_yticks(range(len(DEFENCES))); axes[0].set_yticklabels(DEFENCES,fontsize=9)
    axes[1].set_yticks([])
    sm=plt.cm.ScalarMappable(cmap=reds,norm=plt.Normalize(0,100))
    cb=fig.colorbar(sm,ax=axes,fraction=.030,pad=.02); cb.set_label("Rate (%)",fontsize=9.5)
    fig.savefig(f"{OUT}/fig_utility_cost.png"); plt.close(fig)


def fig_forest(fo):
    fig, ax = plt.subplots(figsize=(W,3.0))
    labs=[f[0] for f in fo][::-1]
    for i,(lab,est,lo,hi,asr) in enumerate(fo[::-1]):
        if lo is not None:
            ax.plot([lo,hi],[i,i],color=DARK,lw=2.6,solid_capstyle="butt")
            for x in (lo,hi): ax.plot([x,x],[i-.16,i+.16],color=DARK,lw=2.6)
            ax.text(hi+1.5,i,f"{est:.1f} pp [{lo:.1f}, {hi:.1f}]",va="center",fontsize=9)
        else:
            ax.text(2.0,i,f"no clear reduction (attack still wins {asr:.1f}% of the time)",va="center",fontsize=9)
        ax.plot(est,i,"o",ms=8,mfc="white",mec=BLUE,mew=2.4,zorder=3)
    ax.axvline(0,color=DARK,ls="--",lw=1.6)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=9.5)
    ax.set_xlim(-3,86); ax.set_xlabel("Reduction in TokenBreak success (percentage points)")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x",ls=":",color="#BBB")
    fig.savefig(f"{OUT}/fig_tokenbreak_reduction.png"); plt.close(fig)


def fig_margin(mg):
    fig, ax = plt.subplots(figsize=(W,3.2))
    cols=["#BBD6EA","#5B9BD5","#1F4E79"]; w=.26; x=np.arange(len(ATTACKS))
    for j,(bl,n) in enumerate(F_BINS):
        v,lo,hi=[],[],[]
        for a in ATTACKS:
            k=mg[a][j]; v.append(100*k/n)
            lo_,hi_=wilson(k,n); lo.append(v[-1]-lo_); hi.append(hi_-v[-1])
        ax.bar(x+(j-1)*w,v,w,yerr=[lo,hi],capsize=3,color=cols[j],edgecolor=DARK,lw=1.0,
               error_kw=dict(lw=1.4,ecolor=DARK),label=f"{bl}, n={n}")
    ax.set_xticks(x); ax.set_xticklabels(ATTACKS,fontsize=9.5)
    ax.set_ylabel("Attack success rate (%)"); ax.set_ylim(0,118); ax.set_yticks(range(0,101,20))
    ax.legend(title="How sure the classifier was before the attack",ncol=3,loc="upper center",
              bbox_to_anchor=(.5,1.17),frameon=False,fontsize=8.8,title_fontsize=8.8)
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",ls=":",color="#BBB")
    fig.savefig(f"{OUT}/fig_margin_stratified.png"); plt.close(fig)


def fig_mechanism(mech):
    order = ["none", "acted", "reencode", "acted+reencode"]
    pretty = {"none":"Did nothing","acted":"Detected or blocked",
              "reencode":"Rewrote the text","acted+reencode":"Both"}
    cols = {"none":GREY,"acted":RED,"reencode":AMBER,"acted+reencode":TEAL}
    fig, ax = plt.subplots(figsize=(W,2.9))
    left = np.zeros(len(DEFENCES))
    for k in order:
        v = np.array([mech[d].get(k,0) for d in DEFENCES], float)
        ax.barh(np.arange(len(DEFENCES))[::-1], v, left=left, color=cols[k],
                edgecolor="white", lw=1.2, label=pretty[k], height=.7)
        left += v
    ax.set_yticks(np.arange(len(DEFENCES))[::-1]); ax.set_yticklabels(DEFENCES, fontsize=8.8)
    ax.set_xlabel("Evaluations (936 per defence)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(.5,1.02), frameon=False, fontsize=8.8)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(f"{OUT}/fig_mechanism.png"); plt.close(fig)


def fig_calibration(cal):
    fig, ax = plt.subplots(figsize=(W*.68,2.8))
    v=[cal["w5_at_floor"], cal["w10_at_floor"]]
    ax.bar(["5-token window\n(as published)","10-token window\n(used here)"],v,
           color=[RED,TEAL],edgecolor=DARK,lw=1.2,width=.55)
    for i,x in enumerate(v):
        ax.text(i,x+.4,f"{x:.2f}%",ha="center",fontsize=10,fontweight="bold")
    ax.set_ylabel("Ordinary clean text stuck at the\nlowest score the filter can give (%)")
    ax.set_ylim(0,max(v)*1.35)
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",ls=":",color="#BBB")
    fig.savefig(f"{OUT}/fig_cpt_calibration.png"); plt.close(fig)


def fig_detection(df):
    """F1 for every attack x defence pair. Makes the 'good at exactly one thing' point visible."""
    amap = {"tokenbreak":"TokenBreak","advtok":"AdvTok","unicode_invisible":"Invisible",
            "unicode_homoglyph":"Homoglyph","unicode_compat":"Compatibility","unicode_reorder":"Reorder"}
    dmap = {"tokenizer_translation":"Tokenizer translation","canonical_reject":"Canonical reject",
            "canonical_replace":"Canonical replace","unicode_sanitise":"Unicode sanitiser",
            "nfkc_confusable":"NFKC + confusables","cpt_global":"Global CPT","cpt_window":"Window CPT"}
    d = df.copy()
    d["a"] = d["attack"].map(amap); d["d"] = d["defence"].map(dmap)
    d["f1"] = pd.to_numeric(d["f1"], errors="coerce").fillna(0.0) * 100
    M = np.zeros((len(DEFENCES), len(ATTACKS)))
    for r in range(len(DEFENCES)):
        for c in range(len(ATTACKS)):
            s = d[(d["d"] == DEFENCES[r]) & (d["a"] == ATTACKS[c])]
            if s.empty: return False
            M[r, c] = float(s.iloc[0]["f1"])
    reds = LinearSegmentedColormap.from_list("r", ["#FFF5F0","#FCAE91","#FB6A4A","#CB181D","#67000D"])
    fig, ax = plt.subplots(figsize=(W, 3.4))
    ax.imshow(M, cmap=reds, vmin=0, vmax=100, aspect="auto")
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.text(c, r, f"{M[r,c]:.0f}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white" if M[r, c] >= 55 else DARK)
    ax.set_xticks(range(len(ATTACKS))); ax.set_xticklabels(ATTACKS, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(DEFENCES))); ax.set_yticklabels(DEFENCES, fontsize=9)
    ax.set_xlabel("Attack"); ax.set_ylabel("Defence")
    ax.set_xticks(np.arange(-.5, len(ATTACKS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(DEFENCES), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.0)
    ax.tick_params(which="minor", length=0); ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap=reds, norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, ax=ax, fraction=.030, pad=.02); cb.set_label("F1 (%)", fontsize=9.5)
    fig.savefig(f"{OUT}/fig_detection_f1.png"); plt.close(fig)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="."); ap.add_argument("--out", default="figures")
    a = ap.parse_args(); REPO, OUT = a.repo, a.out
    os.makedirs(OUT, exist_ok=True)
    print("loading:")
    undef  = table("undefended attacks", lambda: None, F_UNDEF)
    cov    = table("coverage matrix", load_cover, F_COVER)
    margin = table("margin stratification", load_margin, F_MARGIN)
    print("drawing:")
    for fn, nm in ((lambda: fig_undefended(undef), "fig_undefended_asr"),
                   (lambda: fig_coverage(cov), "fig_coverage_matrix"),
                   (lambda: fig_tradeoff(cov, F_PRESERVE), "fig_security_utility_tradeoff"),
                   (lambda: fig_utility(F_BLOCK, F_CHANGE), "fig_utility_cost"),
                   (lambda: fig_forest(F_FOREST), "fig_tokenbreak_reduction"),
                   (lambda: fig_margin(margin), "fig_margin_stratified"),
                   (lambda: fig_calibration(F_CALIB), "fig_cpt_calibration")):
        fn(); print("  " + nm)
    mech = load_mechanism() if pd is not None else None
    if mech: fig_mechanism(mech); print("  fig_mechanism")
    else:    print("  fig_mechanism  SKIPPED - needs data/defence_matrix.csv "
                   "with an attack_mechanism column. Run with --repo.")
    det = load_detection() if pd is not None else None
    if det is not None and fig_detection(det): print("  fig_detection_f1")
    else: print("  fig_detection_f1  SKIPPED - needs results/step10/detection_metrics.csv.")
    if FELLBACK:
        print("\nTranscribed constants used for: " + ", ".join(FELLBACK) +
              "\nRun with --repo /path/to/project to read your real files.")