"""
MGT0000 / ITAG7105 Data Mining Resit Assignment
Complete Reproducible Data Processing & Modeling Pipeline

Fixes applied from original draft:
1. Resolved all syntax and OCR parsing errors (regex strings, typos, missing brackets/quotes).
2. Integrated proper exception handling and directory creation for output files.
3. Structured execution into distinct computational phases corresponding to Sections 3 and 4 of the assignment.
"""

import os
import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ML & NLP Imports
import nltk
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# Download NLTK dependencies
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Set global parameters
DATA_PATH = r"C:/Users/kshit/Downloads/MGT0000_ITAG7105_resit_dataset.csv"
OUTPUT_DIR = r"C:/Users/kshit/airline_project_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42

SERVICE_RATING_COLS = [
    "Seat Comfort",
    "Food & Beverages",
    "Inflight Entertainment",
    "Wifi & Connectivity",
    "Value For Money",
]


# ==========================================
# SECTION 3: DATA PRE-PROCESSING
# ==========================================


def add_service_availability_flags(df, rating_cols=None, fill_value=0.0):
    """Encode missingness in service ratings as explicit availability flags."""
    df = df.copy()
    cols = rating_cols if rating_cols is not None else SERVICE_RATING_COLS
    for col in cols:
        if col not in df.columns:
            continue
        clean_col_name = col.replace(" & ", "_").replace(" ", "_")
        flag_col = f"{clean_col_name}_Available"
        df[flag_col] = df[col].notna().astype(int)
        df[col] = df[col].fillna(fill_value)
    return df


def drop_constant_verified(df):
    """Drop constant 'Verified' column if all values are identical."""
    df = df.copy()
    if "Verified" in df.columns:
        if df["Verified"].nunique() == 1:
            df = df.drop(columns=["Verified"])
    return df


def parse_review_date(df):
    """Parse 'Review Date' into datetime format."""
    df = df.copy()
    if "Review Date" in df.columns:
        cleaned = (
            df["Review Date"]
            .astype(str)
            .str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True)
        )
        df["Review_Date_Parsed"] = pd.to_datetime(
            cleaned, format="%d %B %Y", errors="coerce"
        )
    return df


def clean_categorical_fields(df):
    """Fill missing categorical values with 'Unknown'."""
    df = df.copy()
    for col in ["Aircraft", "Type Of Traveller"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    return df


def parse_route(df):
    """Split Route string into Origin, Destination, and Via layovers."""
    df = df.copy()
    if "Route" in df.columns:
        route = df["Route"].astype(str)
        via_split = route.str.split(" via ", n=1, expand=True)
        main_route = via_split[0]
        df["Route_Via"] = via_split[1] if via_split.shape[1] > 1 else np.nan

        od_split = main_route.str.split(" to ", n=1, expand=True)
        df["Route_Origin"] = od_split[0]
        df["Route_Destination"] = (
            od_split[1] if od_split.shape[1] > 1 else np.nan
        )

        df.loc[
            df["Route"].isna(),
            ["Route_Origin", "Route_Destination", "Route_Via"],
        ] = np.nan
    return df


def parse_date_flown(df):
    """Parse 'Date Flown' string into month-level datetime."""
    df = df.copy()
    if "Date Flown" in df.columns:
        df["Date_Flown_Parsed"] = pd.to_datetime(
            df["Date Flown"], format="%B %Y", errors="coerce"
        )
    return df


def summarise_cleaning_impact(raw_df, clean_df, cols=None):
    """Generate pre/post cleaning metric comparison table."""
    cols = cols if cols is not None else SERVICE_RATING_COLS
    rows = []
    for col in cols:
        rows.append(
            {
                "Column": col,
                "Missing_Before": raw_df[col].isna().sum(),
                "Mean_Before": raw_df[col].mean(),
                "Missing_After": (
                    clean_df[col].isna().sum() if col in clean_df else np.nan
                ),
                "Mean_After_incl_zeros": (
                    clean_df[col].mean() if col in clean_df else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# Text Pipeline Prep
NEGATION_WORDS = {
    "no",
    "not",
    "nor",
    "never",
    "none",
    "nothing",
    "neither",
    "without",
    "n't",
    "cannot",
    "wasn't",
    "isn't",
    "didn't",
    "don't",
    "doesn't",
    "won't",
    "couldn't",
    "wouldn't",
    "shouldn't",
}
STOPWORDS = set(stopwords.words("english")) - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()


def strip_verification_marker(text):
    """Remove header verification labels."""
    text = re.sub(
        r"^(?:trip verified|not verified)\s*\|?\s*",
        "",
        str(text),
        flags=re.IGNORECASE,
    )
    return text.strip()


def clean_text(text):
    """Normalize text and retain contract negations."""
    text = strip_verification_marker(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize tokens."""
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


def preprocess_review_column(df, text_col="Review"):
    """Pipeline driver for text cleaning."""
    df = df.copy()
    df["Review_Clean"] = df[text_col].astype(str).apply(clean_text)
    df["Review_Tokens"] = df["Review_Clean"].apply(tokenize_and_lemmatize)
    df["Review_Token_Count"] = df["Review_Tokens"].apply(len)
    return df


# ==========================================
# MAIN EXECUTION SCRIPT
# ==========================================

if __name__ == "__main__":
    print("=== Step 1: Loading Dataset ===")
    df_raw = pd.read_csv(DATA_PATH)
    print("Raw Shape:", df_raw.shape)

    # Preprocessing
    clean_df = add_service_availability_flags(df_raw)
    clean_df = drop_constant_verified(clean_df)
    clean_df = parse_review_date(clean_df)
    clean_df = clean_categorical_fields(clean_df)
    clean_df = parse_route(clean_df)
    clean_df = parse_date_flown(clean_df)
    clean_df = preprocess_review_column(clean_df)

    summary = summarise_cleaning_impact(df_raw, clean_df)
    print("\n--- Cleaning Summary Impact ---")
    print(summary.to_string(index=False))

    # Add Review Year
    clean_df["Review_Year"] = clean_df["Review_Date_Parsed"].dt.year

    # ==========================================
    # SECTION 4A: NUMERIC EDA & VISUALS
    # ==========================================
    print("\n=== Step 2: Running Numeric EDA & Visualizations ===")
    sns.set_style("whitegrid")

    # Figure 1: Numeric EDA Matrix
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    seat_rec = (
        clean_df.groupby("Seat Type")["Recommended"]
        .apply(lambda x: (x == "yes").mean())
        .sort_values()
    )
    axes[0, 0].barh(seat_rec.index, seat_rec.values, color="steelblue")
    axes[0, 0].set_title("Recommendation Rate by Seat Type")
    axes[0, 0].set_xlabel("Proportion Recommending")

    year_rec = clean_df.groupby("Review_Year")["Recommended"].apply(
        lambda x: (x == "yes").mean()
    )
    axes[0, 1].plot(
        year_rec.index, year_rec.values, marker="o", color="darkred"
    )
    axes[0, 1].set_title("Recommendation Rate by Review Year")
    axes[0, 1].set_ylabel("Proportion Recommending")

    vfm_rated = clean_df[clean_df["Value_For_Money_Available"] == 1][
        "Value For Money"
    ]
    axes[1, 0].hist(
        vfm_rated,
        bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        color="seagreen",
        edgecolor="black",
    )
    axes[1, 0].set_title("Distribution of Value For Money Ratings")
    axes[1, 0].set_xlabel("Rating (1-5)")

    corr = clean_df[SERVICE_RATING_COLS].replace(0, np.nan).corr(method="spearman")
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", ax=axes[1, 1], vmin=0, vmax=1
    )
    axes[1, 1].set_title("Correlation Between Service Ratings")

    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "numeric_eda_1.png")
    plt.savefig(fig1_path, dpi=150)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Executive Management Dashboard Visualization
    fig, ax = plt.subplots(figsize=(11, 6))
    years_valid = year_rec.dropna()
    years_str = years_valid.index.astype(int).astype(str)
    values_pct = years_valid.values * 100
    colors = [
        "#c0392b" if y == "2022" else "#2c7fb8" for y in years_str
    ]

    bars = ax.bar(years_str, values_pct, color=colors, width=0.6)
    for bar, val in zip(bars, values_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val:.0f}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    if "2022" in years_str.values:
        idx_2022 = list(years_str).index("2022")
        ax.annotate(
            "2022: recommendation rate\nnearly halved vs 2019 peak",
            xy=("2022", values_pct[idx_2022]),
            xytext=(0.5, 45),
            textcoords="data",
            fontsize=11,
            color="#c0392b",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
        )

    ax.set_ylabel("% of Customers Who Would Recommend", fontsize=12)
    ax.set_title(
        "Customer Recommendation Rate, Year by Year\n(2022 stands out as a service crisis point)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 60)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "management_dashboard_viz.png")
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"Saved: {fig2_path}")

    # ==========================================
    # SECTION 4A: NUMERIC PREDICTIVE MODELING
    # ==========================================
    print("\n=== Step 3: Training Numeric Predictive Models ===")
    y_num = (clean_df["Recommended"] == "yes").astype(int)

    num_features = SERVICE_RATING_COLS + [
        "Seat_Comfort_Available",
        "Food_Beverages_Available",
        "Inflight_Entertainment_Available",
        "Wifi_Connectivity_Available",
        "Value_For_Money_Available",
    ]
    cat_features = ["Seat Type", "Type Of Traveller"]

    X_num = clean_df[num_features + cat_features].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y_num, test_size=0.2, random_state=RANDOM_SEED, stratify=y_num
    )

    num_prep = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_features,
            ),
        ]
    )

    lr_pipe = Pipeline(
        [
            ("prep", num_prep),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )
    lr_grid = GridSearchCV(
        lr_pipe, {"clf__C": [0.01, 0.1, 1, 10]}, cv=5, scoring="f1"
    )
    lr_grid.fit(X_train, y_train)

    rf_pipe = Pipeline(
        [
            ("prep", num_prep),
            (
                "clf",
                RandomForestClassifier(
                    random_state=RANDOM_SEED, class_weight="balanced"
                ),
            ),
        ]
    )
    rf_grid = GridSearchCV(
        rf_pipe,
        {"clf__n_estimators": [100, 200], "clf__max_depth": [5, 10, None]},
        cv=5,
        scoring="f1",
    )
    rf_grid.fit(X_train, y_train)

    print("\n--- Numeric Models Performance ---")
    for name, model in [
        ("Logistic Regression", lr_grid.best_estimator_),
        ("Random Forest", rf_grid.best_estimator_),
    ]:
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        print(f"\nModel: {name}")
        print("ROC-AUC:", round(roc_auc_score(y_test, probs), 3))
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    # Temporal Stress Testing (<2022 vs 2022)
    print("\n--- Temporal Stress Test (<2022 vs 2022) ---")
    train_mask = clean_df["Review_Year"] < 2022
    test_mask = clean_df["Review_Year"] == 2022

    X_str_train, y_str_train = X_num[train_mask], y_num[train_mask]
    X_str_test, y_str_test = X_num[test_mask], y_num[test_mask]

    lr_stress = Pipeline(
        [
            ("prep", num_prep),
            (
                "clf",
                LogisticRegression(
                    C=lr_grid.best_params_["clf__C"],
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    lr_stress.fit(X_str_train, y_str_train)
    probs_str = lr_stress.predict_proba(X_str_test)[:, 1]
    print(
        "Stress Test LR ROC-AUC:",
        round(roc_auc_score(y_str_test, probs_str), 3),
    )

    # ==========================================
    # SECTION 4B: TEXT ANALYSIS & SENTIMENT
    # ==========================================
    print("\n=== Step 4: Text Sentiment & Modeling ===")
    vader = SentimentIntensityAnalyzer()

    clean_df["VADER_Sentiment"] = clean_df["Review_Clean"].apply(
        lambda t: (
            "positive"
            if vader.polarity_scores(str(t))["compound"] >= 0.05
            else (
                "negative"
                if vader.polarity_scores(str(t))["compound"] <= -0.05
                else "neutral"
            )
        )
    )

    clean_df["TextBlob_Sentiment"] = clean_df["Review_Clean"].apply(
        lambda t: (
            "positive"
            if TextBlob(str(t)).sentiment.polarity > 0.1
            else (
                "negative"
                if TextBlob(str(t)).sentiment.polarity < -0.1
                else "neutral"
            )
        )
    )

    # Figure 3: Text Sentiment Analysis & Contradictions
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    sent_counts = pd.DataFrame(
        {
            "VADER": clean_df["VADER_Sentiment"].value_counts(),
            "TextBlob": clean_df["TextBlob_Sentiment"].value_counts(),
        }
    ).fillna(0)
    sent_counts.plot(
        kind="bar", ax=axes[0, 0], color=["#2c7fb8", "#e67e22"]
    )
    axes[0, 0].set_title("Sentiment Distribution: VADER vs TextBlob")
    axes[0, 0].tick_params(axis="x", rotation=0)

    contradiction_data = pd.DataFrame(
        {
            "VADER": clean_df.groupby("VADER_Sentiment")["Recommended"].apply(
                lambda x: (x == "yes").mean()
            ),
            "TextBlob": clean_df.groupby("TextBlob_Sentiment")[
                "Recommended"
            ].apply(lambda x: (x == "yes").mean()),
        }
    )
    contradiction_data.plot(
        kind="bar", ax=axes[0, 1], color=["#2c7fb8", "#e67e22"]
    )
    axes[0, 1].set_title("Recommendation Rate by Sentiment Class")
    axes[0, 1].tick_params(axis="x", rotation=0)

    vfm_df = clean_df[clean_df["Value_For_Money_Available"] == 1].copy()
    vfm_df["VADER_score"] = vfm_df["Review_Clean"].apply(
        lambda t: vader.polarity_scores(str(t))["compound"]
    )
    axes[1, 0].scatter(
        vfm_df["Value For Money"],
        vfm_df["VADER_score"],
        alpha=0.15,
        s=15,
        color="teal",
    )
    axes[1, 0].set_title("Text Sentiment vs Value For Money")
    axes[1, 0].axhline(0, color="gray", linestyle="--")

    pos_no = len(
        clean_df[
            (clean_df["VADER_Sentiment"] == "positive")
            & (clean_df["Recommended"] == "no")
        ]
    )
    neg_yes = len(
        clean_df[
            (clean_df["VADER_Sentiment"] == "negative")
            & (clean_df["Recommended"] == "yes")
        ]
    )
    aligned = len(clean_df) - pos_no - neg_yes

    axes[1, 1].bar(
        [
            "Pos Text,\nNot Rec",
            "Neg Text,\nRec",
            "Consistent\nAligned",
        ],
        [pos_no, neg_yes, aligned],
        color=["#c0392b", "#e67e22", "#27ae60"],
    )
    axes[1, 1].set_title("Alignment Between Text Sentiment and Recommendation")

    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "text_eda_1.png")
    plt.savefig(fig3_path, dpi=150)
    plt.close()
    print(f"Saved: {fig3_path}")

    # TF-IDF Modeling
    clean_df["Review_Tokens_Str"] = clean_df["Review_Tokens"].apply(
        lambda toks: " ".join(toks)
    )
    X_text = clean_df["Review_Tokens_Str"]
    y_text = (clean_df["Recommended"] == "yes").astype(int)

    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        X_text,
        y_text,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_text,
    )

    tfidf = TfidfVectorizer(
        max_features=3000, min_df=5, max_df=0.8, ngram_range=(1, 2)
    )
    Xt_train_vec = tfidf.fit_transform(Xt_train)
    Xt_test_vec = tfidf.transform(Xt_test)

    lr_text = LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED, C=1
    )
    lr_text.fit(Xt_train_vec, yt_train)

    rf_text = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    rf_text.fit(Xt_train_vec, yt_train)

    print("\n--- Text-Only Models (TF-IDF) Performance ---")
    for name, model in [
        ("LR (Text TF-IDF)", lr_text),
        ("RF (Text TF-IDF)", rf_text),
    ]:
        probs = model.predict_proba(Xt_test_vec)[:, 1]
        print(f"{name} ROC-AUC:", round(roc_auc_score(yt_test, probs), 3))

    # ==========================================
    # SECTION 4C: FEATURE FUSION (EARLY FUSION)
    # ==========================================
    print("\n=== Step 5: Feature Integration (Early Fusion) ===")
    train_idx = Xt_train.index
    test_idx = Xt_test.index

    X_num_train = X_num.loc[train_idx]
    X_num_test = X_num.loc[test_idx]

    X_num_train_vec = num_prep.fit_transform(X_num_train)
    X_num_test_vec = num_prep.transform(X_num_test)

    X_fusion_train = hstack([csr_matrix(X_num_train_vec), Xt_train_vec])
    X_fusion_test = hstack([csr_matrix(X_num_test_vec), Xt_test_vec])

    lr_fusion = LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED, C=1
    )
    lr_fusion.fit(X_fusion_train, yt_train)

    rf_fusion = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    rf_fusion.fit(X_fusion_train, yt_train)

    print("\n--- Fusion Models Performance ---")
    for name, model in [
        ("LR (Early Fusion)", lr_fusion),
        ("RF (Early Fusion)", rf_fusion),
    ]:
        probs = model.predict_proba(X_fusion_test)[:, 1]
        print(f"{name} ROC-AUC:", round(roc_auc_score(yt_test, probs), 3))

    # Feature Importance Breakdown
    n_num_feats = X_num_train_vec.shape[1]
    imp_fusion = rf_fusion.feature_importances_
    print(
        f"\nTotal Importance - Numeric Features: {imp_fusion[:n_num_feats].sum():.3f}"
    )
    print(
        f"Total Importance - Text Features:    {imp_fusion[n_num_feats:].sum():.3f}"
    )

    print("\n=== Script Complete ===")