print("Starting script...", flush=True)
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency,kruskal
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr, mannwhitneyu
from itertools import combinations
#The survey was run in english and spanish. Some  answers needed to be mapped before the analysis. 
Data=pd.read_csv("SoundSurvey.csv",encoding="cp1252")

#print(Data.shape)
#Renaming the columns.
Data = Data.rename(columns={
    "Number": "id",
    "Gender": "gender",
    "Nationality": "nationality",
    "Ocupation": "occupation",
    "Other Occupation": "occupation_other",
    "Level": "level",
    "Years of Experience": "years",
    "Main Field": "field",
    "Bottleneck": "bottleneck",
    "Library": "library",
    "Use of AI": "ai_use",
    "Principal Concerns": "concerns",
    "How does AI work?": "ai_understanding",
    "Use of AI currently": "ai_current",
    "Likert Scale": "text_to_audio",
    "Unnamed: 15": "parametric",
    "Unnamed: 16": "hybrid",
    "Unnamed: 17": "vocal_imitation",
    "Interest in a call": "interview_interest",
})


Data = Data.drop(columns=["Unnamed: 19"]) #No interviews interests. 

## Counting the nationalities
#print (Data["gender"].value_counts())
nationality_map = {
    "United Kingdom": "British",
    "China": "Chinese",
    "Argentina": "Argentinian",
    "Canada": "Canadian",
}
Data["nationality"] = Data["nationality"].str.strip()
Data["nationality_clean"] = Data["nationality"].replace(nationality_map)
Data["nationality_clean"] = Data["nationality"].str.strip().replace(nationality_map)
#print(Data["nationality_clean"].nunique()) 

#Career Stage-rename and mapping for levels. 
years_map = {
    "< 1yr": "<1", "< 1año": "<1", "<1año": "<1",
    "1-3 yrs": "1-3", "1-3 años": "1-3",
    "3-5 yrs": "3-5", "3-5 años": "3-5",
    "5-10 years": "5-10",
    ">10yrs": ">10", ">10 años": ">10",
}
Data["years_clean"] = Data["years"].map(years_map)
#print(Data["years_clean"].value_counts(dropna=False))

stage_map = {
    "<1": "Entry-level",
    "1-3": "Entry-level",
    "3-5": "Mid-career",
    "5-10": "Established",
    ">10": "Established",
}
Data["CareerStage"] = Data["years_clean"].map(stage_map)
#print(Data["CareerStage"].value_counts(dropna=False))

#Counting the levels participants describe themselves.  
Data["levelC"] = Data["level"].replace({"Novice": "Beginner"})
Data["levelC"] = pd.Categorical(
    Data["levelC"],
    categories=["Beginner", "Intermediate", "Advanced", "Expert"],
    ordered=True,
)
#print(Data["levelC"].value_counts(sort=False))


# Media Type
Data["linear"] = Data["field"].str.contains("Cinema|Series|Podcast|Audio books", na=False)
Data["games"] = Data["field"].str.contains("Games", na=False)
conditions = [
    Data["games"],    # anyone who works in games (including those who also do linear)
    Data["linear"],   # linear media only
]
choices = ["Interactive (games)", "Linear media"] 

Data["media_group"] = np.select(conditions, choices, default="Music and other")#if it doesnt choose between the conditions the user works in music
#print(Data["media_group"].value_counts())

#Library Use 

Data["library_clean"] = Data["library"].str.replace("\xa0", " ").str.strip()

library_map = {
    "No, I've got my own library of sounds.": "Own library",
    "No, tengo mi propia libreria de sonidos.": "Own library",
    "No, I prefer to record samples.": "Records own sounds",
    "No, prefiero grabar mis sonidos.": "Records own sounds",
    "Yes, I often use generative audio to create samples for my sound designs.": "Generative AI",
    "Si, uso la IA para crear sonidos que utilizaré para mis diseños.": "Generative AI",
    "Yes, I used procedural audio to avoid repetitive sounds.": "Procedural audio",
} 
Data["library_clean"] = Data["library_clean"].map(library_map)
#print(Data["library_clean"].value_counts(dropna=False))

#Use of AI 
#print(Data["ai_use"].value_counts())
#print(Data["ai_understanding"].value_counts())

# Analysing the open questions of the survey

def TextCorrection(series):
    return (series.fillna("").astype(str).str.strip()
            .str.encode("ascii", "ignore").str.decode("ascii")
            .str.replace(r"\s+", " ", regex=True))


def count_codes(file_name, text_col, survey_col, fix=None):
    try:
        codes = pd.read_csv(file_name, encoding="utf-8")
    except UnicodeDecodeError:
        codes = pd.read_csv(file_name, encoding="cp1252")
    while len(codes) > 0 and codes.iloc[-1].isna().all():
        codes = codes.iloc[:-1]
    if len(codes) != len(Data):
        print(f"Warning: {file_name} has {len(codes)} rows, but the survey has {len(Data)}")

    n = min(len(codes), len(Data))
    coded_text = TextCorrection(codes[text_col]).values[:n]
    survey_text = TextCorrection(Data[survey_col]).values[:n]
    if not (coded_text == survey_text).all():
        print("Warning: rows don't match the survey order in", file_name)

    codes["id"] = range(1, len(codes) + 1)

    code_cols = [c for c in codes.columns if c.startswith("Code")]
    for col in code_cols:
        codes[col] = codes[col].str.strip()
        if fix:
            codes[col] = codes[col].replace(fix)

    long = codes.melt(id_vars="id", value_vars=code_cols, value_name="code")
    long = long.dropna(subset=["code"])

    n_answered = codes[code_cols[0]].notna().sum()
    counts = long["code"].value_counts()
    print("\n==", file_name, "| participants who answered:", n_answered)
    print(pd.DataFrame({"n": counts, "%": (counts / n_answered * 100).round(1)}))
    return long


# 1. Bottleneck
bottleneck_fix = {"mixing": "Mixing", "Accesability to studios": "Access to studios"}
long_bottleneck = count_codes("Bottleneck.csv", "Bottleneck", "bottleneck", fix=bottleneck_fix)

# 2. Principal concerns
long_concerns = count_codes("PrincipalConcerns.csv", "Principal Concerns", "concerns")

# 3. Use of AI
long_ai = count_codes("UseOfAI.csv", "Use of AI currently", "ai_current")


def CorrectionValue(pvals):
    p = np.asarray(pvals)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1, running)
    return adjusted.round(3)

def compare_by_group(long, group_col, codes_to_test):
    wide = pd.crosstab(long["id"], long["code"]).clip(upper=1).reset_index()
    df = Data[["id", group_col]].merge(wide, on="id", how="inner")

    rows = []
    for code in codes_to_test:
        table = pd.crosstab(df[group_col], df[code])
        chi2, p, dof, expected = chi2_contingency(table)
        n = table.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
        pct = (df.groupby(group_col)[code].mean() * 100).round(1)
        rows.append({"code": code, **pct.to_dict(), "chi2": round(chi2, 2),
                     "p": round(p, 3), "cramers_v": round(cramers_v, 2),
                     "min_expected": round(expected.min(), 1)})

    results = pd.DataFrame(rows)
    results["p_CorrectionValue"] = CorrectionValue(results["p"])
    return results

##likert Scale- Question
options = ["text_to_audio", "parametric", "hybrid", "vocal_imitation"]
for col in options:
    Data[col] = Data[col].str.replace("Option ", "").astype(int)

# Percentage of each rating, 1 (least preferred) to 5 (most preferred)
dist = pd.DataFrame({c: Data[c].value_counts(normalize=True).sort_index() * 100 for c in options}).T
print(dist.round(1))
#Does experience differ in the option selection? 
for group in ["CareerStage", "media_group"]:
    rows = []
    for c in options:
        groups = [g[c].values for _, g in Data.groupby(group)]
        H, p = kruskal(*groups)
        eps2 = (H - len(groups) + 1) / (len(Data) - len(groups))   # effect size
        rows.append({"item": c, "H": round(H, 2), "p": p, "epsilon2": round(eps2, 3)})
    res = pd.DataFrame(rows)
    res["p_CorrectionValue"] = CorrectionValue(res["p"])

# Multiple choice visualization
labels = {"text_to_audio": "Text-to-audio", "parametric": "Parametric",
          "hybrid": "Hybrid", "vocal_imitation": "Vocal imitation"}

pct = pd.DataFrame({c: Data[c].value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0) * 100
                    for c in options}).T
pct = pct.loc[pct[[4, 5]].sum(axis=1).sort_values().index]   # most preferred at the top
colors = {1: "#b2182b", 2: "#ef8a62", 3: "#bdbdbd", 4: "#67a9cf", 5: "#2166ac"}
names = {1: "1 (least preferred)", 2: "2", 3: "3", 4: "4", 5: "5 (most preferred)"}

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(6.5, 2.6))
y = np.arange(len(pct))
left = -(pct[1] + pct[2] + pct[3] / 2) 
for r in [1, 2, 3, 4, 5]:
    ax.barh(y, pct[r], left=left, color=colors[r], label=names[r], edgecolor="white", height=0.6)
    left = left + pct[r]
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y, [labels[c] for c in pct.index])
ax.set_xlabel("Percentage of participants")
ax.set_xlim(-70, 70)
ax.set_xticks(range(-60, 61, 20), [f"{abs(t)}%" for t in range(-60, 61, 20)])
ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/likert_diverging.jpg", bbox_inches="tight")


##COMPARISON OF YEARS OF EXPERIENCE AND MEDIA TYPE.
##We want to understand if there is sth interesting in the use of AI on how people 
def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": "#333333",
        "axes.edgecolor": "#BBBBBB",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "grid.color": "#E6E6E6",
        "axes.axisbelow": True,
        "xtick.color": "#555555",
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "savefig.bbox": "tight",
    })

set_style()
os.makedirs("figures", exist_ok=True)

STAGES = ["Entry-level", "Mid-career", "Established"]

MEDIA = ["Linear media", "Interactive (games)", "Music and other"]
AI_COLORS = {"Yes": "#A23B72", "Maybe": "#F4A6A6", "No": "#D9D9D9"}
top_bottlenecks = long_bottleneck["code"].value_counts().head(4).index.tolist()
top_concerns = long_concerns["code"].value_counts().head(4).index.tolist()
STAGE_COLORS = ["#F4A6A6", "#E0607E", "#4A1E5C"]   


def get_pct(long, group_col="CareerStage", order=STAGES, top=6):
    """Percentage of participants in each group who mentioned each code."""
    wide = pd.crosstab(long["id"], long["code"]).clip(upper=1).reset_index()
    df = Data[["id", group_col]].merge(wide, on="id", how="inner")
    top_codes = long["code"].value_counts().head(top).index.tolist()
    pct = df.groupby(group_col)[top_codes].mean().reindex(order) * 100
    sizes = df[group_col].value_counts().reindex(order)
    return pct, sizes, top_codes

#Career and type of media. Are they correlated?
overlap = pd.crosstab(Data["CareerStage"], Data["media_group"])
print(overlap)
chi2, p, dof, expected = chi2_contingency(overlap)
print(f"Career stage vs media type: chi2({dof}) = {chi2:.2f}, p = {p:.3f}")

# % who would use AI, in each career stage x media type combination
yes_pct = pd.crosstab(Data["CareerStage"], Data["media_group"],
                      values=(Data["ai_use"] == "Yes"), aggfunc="mean") * 100
counts = pd.crosstab(Data["CareerStage"], Data["media_group"])

print("% saying yes to AI:")
print(yes_pct.round(0))
print("\nNumber of people in each group:")
print(counts)

# Median text-to-audio rating in each combination
print("\nMedian text-to-audio rating:")
print(Data.pivot_table(index="CareerStage", columns="media_group",
                       values="text_to_audio", aggfunc="median"))


# ===================== COMBINED FIGURES: CAREER STAGE + MEDIA TYPE =====================
MEDIA_COLORS = ["#9ECAE1", "#4292C6", "#08519C"]   # blues, so media type looks different from career stage

# Each grouping: (column, group order, panel label, colours)
GROUPINGS = [("CareerStage", STAGES, "By career stage", STAGE_COLORS),
             ("media_group", MEDIA, "By media type", MEDIA_COLORS)]


# ---------- Donuts: career stage on top, media type below ----------
def donuts_combined(col, colors, title, filename, groupings=GROUPINGS, min_label=5,
                    text_colors=None):
    cats = list(colors.keys())
    text_colors = text_colors or {}
    ncols = max(len(order) for _, order, _, _ in groupings)
    fig, axes = plt.subplots(len(groupings), ncols, figsize=(2.1 * ncols, 2.6 * len(groupings)))
    for row, (group_col, order, row_label, _) in enumerate(groupings):
        for k, grp in enumerate(order):
            ax = axes[row, k]
            sub = Data.loc[Data[group_col] == grp, col].dropna()
            counts = sub.value_counts().reindex(cats, fill_value=0)
            shares = counts / counts.sum() * 100
            wedges, _ = ax.pie(counts, colors=[colors[c] for c in cats], startangle=90,
                               counterclock=False,
                               wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.5))
            for w, cat in zip(wedges, cats):
                if cat == cats[0] or shares[cat] < min_label:
                    continue
                ang = np.deg2rad((w.theta1 + w.theta2) / 2)
                ax.text(0.81 * np.cos(ang), 0.81 * np.sin(ang), f"{shares[cat]:.0f}%",
                        ha="center", va="center", fontsize=7.5,
                        color=text_colors.get(cat, "#333333"))
            ax.text(0, 0, f"{shares.iloc[0]:.0f}%\n{cats[0].lower()}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=colors[cats[0]], linespacing=1.1)
            ax.set_title(f"{grp}\n(n={len(sub)})", fontsize=8.5, loc="center", fontweight="normal")
        for k in range(len(order), ncols):   # hide empty spaces if a row has fewer groups
            axes[row, k].axis("off")
        axes[row, 0].annotate(row_label, xy=(-0.25, 0.5), xycoords="axes fraction",
                              rotation=90, ha="center", va="center", fontsize=9.5,
                              fontweight="bold", color="#555555")
    fig.legend(wedges, cats, loc="lower center", ncol=len(cats), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0.04, 1, 0.96))
    fig.savefig(filename)


# ---------- Heatmaps side by side ----------
def heatmap_combined(long, title, filename, groupings=GROUPINGS, top=6):
    top_codes = long["code"].value_counts().head(top).index.tolist()   # same rows in both panels
    cmap = LinearSegmentedColormap.from_list("plum", ["#FFF5F7", "#F4A6A6", "#A23B72", "#4A1E5C"])
    widths = [len(order) for _, order, _, _ in groupings]
    fig, axes = plt.subplots(1, len(groupings), figsize=(1.35 * sum(widths) + 2.2, 0.5 * top + 1.6),
                             gridspec_kw={"width_ratios": widths, "wspace": 0.06}, sharey=True)
    for ax, (group_col, order, panel_label, _) in zip(axes, groupings):
        pct, sizes, _ = get_pct(long, group_col=group_col, order=order, top=top)
        data = pct[top_codes].T.values
        ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=60)   # same colour scale in both
        ax.set_xticks(range(len(order)), [f"{g}\n(n={sizes[g]})" for g in order], fontsize=7.5)
        ax.set_yticks(range(len(top_codes)), top_codes)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if v > 35 else "#333333")
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.tick_top()
        ax.set_title(panel_label, fontsize=9.5, color="#555555", loc="center", pad=34)
    fig.suptitle(title, x=0.02, y=1.08, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(filename)


#BUBBLE Gird: Use of AI
def bubbles_combined(long, title, filename, groupings=GROUPINGS, top=5):
    top_codes = long["code"].value_counts().head(top).index.tolist()
    widths = [len(order) for _, order, _, _ in groupings]
    fig, axes = plt.subplots(1, len(groupings), figsize=(1.25 * sum(widths) + 2.2, 0.6 * top + 1.8),gridspec_kw={"width_ratios": widths, "wspace": 0.05}, sharey=True)
    for ax, (group_col, order, panel_label, colors) in zip(axes, groupings):
        pct, sizes, _ = get_pct(long, group_col=group_col, order=order, top=top)
        for j, (grp, color) in enumerate(zip(order, colors)):
            for i, code in enumerate(top_codes):
                v = pct.loc[grp, code]
                yi = len(top_codes) - 1 - i
                ax.scatter(j, yi, s=v * 22 + 8, color=color, alpha=0.9,
                           edgecolor="white", linewidth=1)
                ax.text(j, yi - 0.38, f"{v:.0f}%", ha="center", va="top",
                        fontsize=7, color="#555555")
        ax.set_xticks(range(len(order)), [f"{g}\n(n={sizes[g]})" for g in order], fontsize=7.5)
        ax.set_yticks(range(len(top_codes)), top_codes[::-1])
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(-0.9, len(top_codes) - 0.4)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.tick_top()
        ax.set_title(panel_label, fontsize=9.5, color="#555555", loc="center", pad=34)
    fig.suptitle(title, x=0.02, y=1.08, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(filename)


donuts_combined("ai_use", AI_COLORS, "Would you use AI in your workflow?","figures/ai_use_combined.jpg")
heatmap_combined(long_bottleneck, "Bottlenecks", "figures/bottlenecks_combined.jpg")
bubbles_combined(long_concerns, "Concerns about AI", "figures/concerns_combined.jpg")

#  STATISTICS: CAREER STAGE vs MEDIA TYPE, SIDE BY SIDE 
def cramers_v(chi2, table):
    return np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))


def test_openness(group_col):
    table = pd.crosstab(Data[group_col], Data["ai_use"])
    chi2, p, dof, expected = chi2_contingency(table)
    return [{"Question": "Openness to AI", "Item": "Yes / Maybe / No",
             "Stat": f"chi2({dof})={chi2:.2f}", "p": p, "p_adj": p,   # single test: no correction
             "Effect": f"V={cramers_v(chi2, table):.2f}", "min_exp": expected.min()}]


def test_ratings(group_col):
    rows = []
    for c in options:
        groups = [g[c].values for _, g in Data.groupby(group_col)]
        H, p = kruskal(*groups)
        eps2 = max(0, (H - len(groups) + 1) / (len(Data) - len(groups)))   # negative values -> 0
        rows.append({"Question": "Tool ratings", "Item": labels[c],
                     "Stat": f"H({len(groups) - 1})={H:.2f}", "p": p,
                     "Effect": f"eps2={eps2:.2f}", "min_exp": np.nan})
    adj = CorrectionValue([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_adj"] = a
    return rows


def test_codes(long, question, group_col, codes):
    wide = pd.crosstab(long["id"], long["code"]).clip(upper=1).reset_index()
    df = Data[["id", group_col]].merge(wide, on="id", how="inner")
    rows = []
    for code in codes:
        table = pd.crosstab(df[group_col], df[code])
        chi2, p, dof, expected = chi2_contingency(table)
        rows.append({"Question": question, "Item": code,
                     "Stat": f"chi2({dof})={chi2:.2f}", "p": p,
                     "Effect": f"V={cramers_v(chi2, table):.2f}", "min_exp": expected.min()})
    adj = CorrectionValue([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_adj"] = a
    return rows


def all_tests(group_col):
    rows = (test_openness(group_col)
            + test_ratings(group_col)
            + test_codes(long_bottleneck, "Bottlenecks", group_col, top_bottlenecks)
            + test_codes(long_concerns, "Concerns", group_col, top_concerns))
    out = pd.DataFrame(rows)
    out["p_adj"] = out["p_adj"].round(3)
    out["Sig"] = np.where(out["p_adj"] < 0.05, "*", "")          # significant after correction
    out["Note"] = np.where(out["min_exp"] < 5, "low exp.", "")    # chi-square only approximate
    return out.set_index(["Question", "Item"])[["Stat", "p_adj", "Effect", "Sig", "Note"]]


side_by_side = pd.concat({"Career stage": all_tests("CareerStage"),"Media type": all_tests("media_group")}, axis=1)
print(side_by_side.to_string())
side_by_side.to_csv("stats_side_by_side.csv")   # open in Excel to read it comfortably

