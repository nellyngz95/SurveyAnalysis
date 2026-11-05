print("Starting script...", flush=True)
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency,kruskal

#The survey was run in english and spanish. Some  answers needed to be mapped before the analysis. 
Data=pd.read_csv("SoundSurvey.csv",encoding="cp1252")

print(Data.shape)

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


Data = Data.drop(columns=["Unnamed: 19"])



## COUNTING GENDER AND NATIONALITY
print (Data["gender"].value_counts())
nationality_map = {
    "United Kingdom": "British",
    "China": "Chinese",
    "Argentina": "Argentinian",
    "Canada": "Canadian",
}
Data["nationality"] = Data["nationality"].str.strip()
Data["nationality_clean"] = Data["nationality"].replace(nationality_map)



Data["nationality_clean"] = Data["nationality"].str.strip().replace(nationality_map)
print(Data["nationality_clean"].nunique()) 

#Career Stage:
years_map = {
    "< 1yr": "<1", "< 1año": "<1", "<1año": "<1",
    "1-3 yrs": "1-3", "1-3 años": "1-3",
    "3-5 yrs": "3-5", "3-5 años": "3-5",
    "5-10 years": "5-10",
    ">10yrs": ">10", ">10 años": ">10",
}
Data["years_clean"] = Data["years"].map(years_map)
print(Data["years_clean"].value_counts(dropna=False))

stage_map = {
    "<1": "Entry-level",
    "1-3": "Mid-career",
    "3-5": "Mid-career",
    "5-10": "Senior",
    ">10": "Established",
}
Data["career_stage3"] = Data["years_clean"].map(stage_map)
print(Data["career_stage3"].value_counts(dropna=False))

#Level 
Data["levelC"] = Data["level"].replace({"Novice": "Beginner"})
Data["levelC"] = pd.Categorical(
    Data["levelC"],
    categories=["Beginner", "Intermediate", "Advanced", "Expert"],
    ordered=True,
)
print(Data["levelC"].value_counts(sort=False))


# Media Type
Data["linear"] = Data["field"].str.contains("Cinema|Series|Podcast|Audio books", na=False)
Data["games"] = Data["field"].str.contains("Games", na=False)
conditions = [
    Data["games"],    # anyone who works in games (including those who also do linear)
    Data["linear"],   # linear media only
]
choices = ["Interactive (games)", "Linear media"] 

Data["media_group"] = np.select(conditions, choices, default="Music and other")#if it doesnt choose between the conditions the user works in music
print(Data["media_group"].value_counts())

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
print(Data["library_clean"].value_counts(dropna=False))

#Use of AI 
print(Data["ai_use"].value_counts())
print(Data["ai_understanding"].value_counts())

#Analysisng the oopen questions of the survey. 
def count_codes(file_name, text_col, survey_col, fix=None):
    codes = pd.read_csv(file_name, encoding="cp1252") #avoiding "ñ"
    
    coded_text = codes[text_col].fillna("").str.strip().values
    survey_text = Data[survey_col].fillna("").str.strip().values[:len(codes)]
    if not (coded_text == survey_text).all():
        print("Warning: rows don't match the survey order in", file_name)

    codes["id"] = range(1, len(codes) + 1)

    # Use every column whose name starts with "Code" (Code, Code2, Code3...)
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


def holm(pvals):
    """Holm correction: adjusts p-values when you run several tests at once."""
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
    # One row per participant: 1 if they mentioned the code, 0 if not
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
    results["p_holm"] = holm(results["p"])
    return results

print(compare_by_group(long_bottleneck, "career_stage3",
      ["Searching for sounds", "Creating sounds", "Editing and tracklaying"]))

##likert Scale
options = ["text_to_audio", "parametric", "hybrid", "vocal_imitation"]
for col in options:
    Data[col] = Data[col].str.replace("Option ", "").astype(int)

# Median and interquartile range (Q1 to Q3)
print(Data[options].describe().loc[["50%", "25%", "75%"]])

# Percentage of each rating, 1 (least preferred) to 5 (most preferred)
dist = pd.DataFrame({c: Data[c].value_counts(normalize=True).sort_index() * 100 for c in options}).T
print(dist.round(1))
#Does experience differ in the option selection? 
for group in ["career_stage3", "media_group"]:
    rows = []
    for c in options:
        groups = [g[c].values for _, g in Data.groupby(group)]
        H, p = kruskal(*groups)
        eps2 = (H - len(groups) + 1) / (len(Data) - len(groups))   # effect size
        rows.append({"item": c, "H": round(H, 2), "p": p, "epsilon2": round(eps2, 3)})
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p"])
    print("\n", group)
    print(Data.groupby(group)[options].median())
    print(res.round(3))

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
left = -(pct[1] + pct[2] + pct[3] / 2)   # centres the neutral answer on zero
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
fig.savefig("figures/likert_diverging.pdf", bbox_inches="tight")

#Open question visualizations
def plot_codes(long, title, filename, top=8):
    n_answered = long["id"].nunique()   # participants with at least one code
    pct = (long["code"].value_counts().head(top) / n_answered * 100)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(pct.index, pct.values, color="#2166ac")
    for i, v in enumerate(pct.values):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_xlabel("Percentage of participants")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")

plot_codes(long_bottleneck, "Main workflow bottlenecks", "figures/bottlenecks.pdf")
plot_codes(long_concerns, "Principal concerns about AI", "figures/concerns.pdf")
plot_codes(long_ai, "Desired AI support", "figures/desired_ai.pdf")