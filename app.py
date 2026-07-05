from __future__ import annotations

import pathlib
import pickle
import sys
import warnings

import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Amazon Recommender", layout="wide")

APP_DIR = pathlib.Path(__file__).resolve().parent
RESOURCES_DIR = APP_DIR / "resources"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from notebooks.helper import (  # noqa: E402
    calculate_bert_content_similarity,
    calculate_sentiment_for_items,
    calculate_xgboost,
    get_ncf_predictions,
    get_svd_predictions_for_user_history,
)


warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


FILES = {
    "10___raw_data.csv": "https://drive.google.com/uc?id=11VltwluUJR87OvO9v-E6ZRebMTNTq-ge",
    "bert_asins.csv": "https://drive.google.com/uc?id=1hZNdi7EjCyQsdo5USDyMxdSNWQbW5gfb",
    "bert_embeddings.npy": "https://drive.google.com/uc?id=1l5_VKscgAulltGSgZqbwFjqNhB86g-KH",
    "ncf_model.pt": "https://drive.google.com/uc?id=1rDBZp7t-XsJlpp6Kj4upuO2RAkMxaxql",
    "trained_svd_model.pkl": "https://drive.google.com/uc?id=1qoL7nBaIQqTimBZStVld8rCr0oekqd9m",
    "user_item_mappings.pkl": "https://drive.google.com/uc?id=1WipR3wt_XIwsydaSHvlGyEt_rHlglyak",
    "xgboost_model.pkl": "https://drive.google.com/uc?id=1XI-yQ-wu8CSgZhQM91GgH3Ny3XdP0Ixz",
}


def resource_is_ready(path: pathlib.Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def download_file(filename: str, url: str, target_path: pathlib.Path) -> None:
    tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    result = gdown.download(url, str(tmp_path), quiet=True, fuzzy=True)
    if result is None or not resource_is_ready(tmp_path):
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Could not download {filename}. Check that the Google Drive file is public "
            "and has not exceeded its download quota."
        )

    tmp_path.replace(target_path)


@st.cache_resource(show_spinner=False)
def ensure_all_files() -> str:
    RESOURCES_DIR.mkdir(exist_ok=True)
    for filename, url in FILES.items():
        path = RESOURCES_DIR / filename
        if not resource_is_ready(path):
            download_file(filename, url, path)
    return str(RESOURCES_DIR)


@st.cache_resource(show_spinner=False)
def load_assets(resources_dir: str):
    resources_path = pathlib.Path(resources_dir)
    df = pd.read_csv(resources_path / "10___raw_data.csv", low_memory=False).rename(
        columns={
            "reviewerID": "user_id",
            "asin": "item_id",
            "reviewText": "review_text",
        }
    )

    bert_vectors = np.load(resources_path / "bert_embeddings.npy", mmap_mode="r")
    bert_asins = pd.read_csv(resources_path / "bert_asins.csv")["asin"].tolist()
    bert_item_id_to_idx = {asin: i for i, asin in enumerate(bert_asins)}

    with open(resources_path / "xgboost_model.pkl", "rb") as file:
        xgb_model = pickle.load(file)

    return df, bert_vectors, bert_item_id_to_idx, xgb_model


def scale_to_0_5(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if values.empty:
        return values

    mean_val = values.mean()
    std_val = values.std()
    if not np.isfinite(std_val) or std_val == 0:
        return pd.Series(2.5, index=values.index)

    standardized = (values - mean_val) / std_val
    min_val = standardized.min()
    max_val = standardized.max()
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val == min_val:
        return pd.Series(2.5, index=values.index)

    return 5 * (standardized - min_val) / (max_val - min_val)


def build_recommendations(
    user_id: str,
    df: pd.DataFrame,
    resources_dir: str,
    bert_vectors: np.ndarray,
    bert_item_id_to_idx: dict[str, int],
    xgb_model,
) -> pd.DataFrame:
    resources_path = pathlib.Path(resources_dir)

    svd_recommended_items_df = get_svd_predictions_for_user_history(
        user_id,
        df,
        df,
        str(resources_path / "trained_svd_model.pkl"),
        n=10,
    )
    if svd_recommended_items_df.empty:
        return svd_recommended_items_df

    sentiment_df = calculate_sentiment_for_items(df, svd_recommended_items_df)
    bert_similarity_df = calculate_bert_content_similarity(
        df,
        sentiment_df,
        user_id,
        bert_vectors,
        bert_item_id_to_idx,
    )

    user_avg_rating = df[df["user_id"] == user_id]["overall"].mean()
    bert_similarity_df["user_ave_rating"] = 0.0 if pd.isna(user_avg_rating) else user_avg_rating
    bert_similarity_df["product_ave_rating"] = [
        df[df["item_id"] == item_id]["overall"].mean()
        for item_id in bert_similarity_df["item_id"].tolist()
    ]
    bert_similarity_df["product_ave_rating"] = bert_similarity_df["product_ave_rating"].fillna(0.0)

    xgb_predictions_df = calculate_xgboost(bert_similarity_df, xgb_model)
    user_recs = get_ncf_predictions(
        xgb_predictions_df,
        str(resources_path / "ncf_model.pt"),
        str(resources_path / "user_item_mappings.pkl"),
        embedding_dim=64,
    )

    if user_recs.empty:
        return user_recs

    user_recs = user_recs.rename(
        columns={
            "svd_rating": "SVD",
            "bert_similarity": "BERT",
            "sentiment_score": "Sentiment",
            "xgb_pred_score": "XGBoost",
            "ncf_score": "NCF",
        }
    )

    for column in ["XGBoost", "NCF", "BERT", "SVD", "Sentiment"]:
        if column in user_recs.columns:
            user_recs[f"{column}_scaled"] = scale_to_0_5(user_recs[column])

    required_scaled_columns = [
        "XGBoost_scaled",
        "NCF_scaled",
        "BERT_scaled",
        "SVD_scaled",
        "Sentiment_scaled",
    ]
    if all(column in user_recs.columns for column in required_scaled_columns):
        user_recs["Hybrid"] = (
            0.30 * user_recs["XGBoost_scaled"]
            + 0.25 * user_recs["NCF_scaled"]
            + 0.15 * user_recs["BERT_scaled"]
            + 0.15 * user_recs["SVD_scaled"]
            + 0.15 * user_recs["Sentiment_scaled"]
        )

    return user_recs


def render_recommendations(top_recs: pd.DataFrame, model_choice: str) -> None:
    for _, row in top_recs.iterrows():
        st.markdown(f"**Product ID: {row['item_id']}**")
        st.write(f"Hybrid Score: {row.get('Hybrid', 0):.5f}")
        st.write(f"SVD: {row.get('SVD', 0):.5f}")
        st.write(f"Sentiment: {row.get('Sentiment', 0):.5f}")
        st.write(f"BERT: {row.get('BERT', 0):.5f}")
        st.write(f"XGBoost: {row.get('XGBoost', 0):.5f}")
        st.write(f"NCF: {row.get('NCF', 0):.5f}")
        st.divider()

    st.subheader("Score Comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(top_recs["item_id"], top_recs[model_choice], color="#5B8DEF")
    ax.set_title(f"{model_choice} Scores for Top 5 Products")
    ax.set_xlabel("Product ID")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.title("Amazon Product Recommender")
    st.caption("Hybrid recommendations using SVD, BERT similarity, sentiment, XGBoost, and NCF.")

    try:
        with st.spinner("Preparing model and data files..."):
            resources_dir = ensure_all_files()
        with st.spinner("Loading data and models..."):
            df, bert_vectors, bert_item_id_to_idx, xgb_model = load_assets(resources_dir)
    except Exception as exc:
        st.error("The app could not prepare its model or data files.")
        st.exception(exc)
        st.info(
            "On Hugging Face Spaces, verify that every Google Drive file is publicly accessible "
            "and that the Space has enough persistent storage for the downloaded resources."
        )
        st.stop()

    st.sidebar.header("Filter Options")
    min_rating = st.sidebar.slider("Minimum Score", 0.0, 5.0, 3.5)
    if "category" in df.columns:
        category_options = sorted(str(value) for value in df["category"].dropna().unique())
        selected_category = st.sidebar.selectbox("Category", ["All"] + category_options)
    else:
        selected_category = "All"

    filtered_df = df
    if selected_category != "All" and "category" in df.columns:
        filtered_df = df[df["category"].astype(str) == selected_category]

    user_id = st.text_input("Enter User ID", value="AAP7PPBU72QFM")
    model_choice = st.selectbox(
        "Choose Model",
        ["Hybrid", "SVD", "BERT", "Sentiment", "XGBoost", "NCF"],
    )

    if not st.button("Get Recommendations", type="primary"):
        return

    user_id = user_id.strip()
    if not user_id:
        st.warning("Enter a user ID to generate recommendations.")
        return

    try:
        with st.spinner("Generating recommendations..."):
            user_recs = build_recommendations(
                user_id,
                filtered_df,
                resources_dir,
                bert_vectors,
                bert_item_id_to_idx,
                xgb_model,
            )
    except Exception as exc:
        st.error(f"Could not generate recommendations for user {user_id}.")
        st.exception(exc)
        return

    if user_recs.empty:
        st.warning(f"No recommendations found for user {user_id}.")
        return

    if model_choice not in user_recs.columns:
        st.warning(f"{model_choice} scores are not available for this result set.")
        return

    filtered_recs = user_recs[user_recs[model_choice] >= min_rating]
    if filtered_recs.empty:
        st.write(f"No recommendations found with {model_choice} scores >= {min_rating}.")
        return

    top_recs = filtered_recs.sort_values(model_choice, ascending=False).head(5)
    st.subheader(f"Top {len(top_recs)} Recommendations for User {user_id}")
    st.markdown(
        f"**Min: {filtered_recs[model_choice].min():.5f}, "
        f"Max: {filtered_recs[model_choice].max():.5f}**"
    )
    render_recommendations(top_recs, model_choice)


if __name__ == "__main__":
    main()
