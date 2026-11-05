import pandas as pd
import numpy as np

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
    "Likert Scale": "likert_1",
    "Unnamed: 15": "likert_2",
    "Unnamed: 16": "likert_3",
    "Unnamed: 17": "likert_4",
    "Interest in a call": "interview_interest",
})


Data = Data.drop(columns=["Unnamed: 19"])

likert_cols = ["likert_1", "likert_2", "likert_3", "likert_4"]
for col in likert_cols:
    Data[col] = Data[col].str.replace("Option ", "").astype(int)

#Analysisng the oopen questions of the survey. 
def count_codes(file_name, text_col, survey_col, fix=None):
    codes = pd.read_csv(file_name, encoding="cp1252")

   
    coded_text = codes[text_col].fillna("").str.strip().values
    survey_text = Data[survey_col].fillna("").str.strip().values[:len(codes)]
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
