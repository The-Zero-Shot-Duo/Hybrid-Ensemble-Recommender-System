import os, sys, pickle, warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gdown

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from notebooks.helper import (
    load_reviews_df,
    get_svd_predictions_for_user_history,
    calculate_sentiment_for_items,
    calculate_bert_content_similarity,
    get_ncf_predictions,
    calculate_xgboost
)

# ---------------------- #
# 1. Basic Configuration
# ---------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

st.set_page_config(page_title="Amazon Recommender", page_icon="📚")
st.title("📚 Amazon Product Recommender")
st.markdown("Choose a user or product to get recommendations. This app uses the final hybrid model outputs.")

# ---------------------- #
# 2. File Download & Check
# ---------------------- #
FILES = {
    "10_%_raw_data.csv": "https://drive.google.com/uc?id=11VltwluUJR87OvO9v-E6ZRebMTNTq-ge",
    "bert_asins.csv": "https://drive.google.com/uc?id=1hZNdi7EjCyQsdo5USDyMxdSNWQbW5gfb",
    "bert_embeddings.npy": "https://drive.google.com/uc?id=1l5_VKscgAulltGSgZqbwFjqNhB86g-KH",
    "ncf_model.pt": "https://drive.google.com/uc?id=1rDBZp7t-XsJlpp6Kj4upuO2RAkMxaxql",
    "trained_svd_model.pkl": "https://drive.google.com/uc?id=1qoL7nBaIQqTimBZStVld8rCr0oekqd9m",
    "user_item_mappings.pkl": "https://drive.google.com/uc?id=1WipR3wt_XIwsydaSHvlGyEt_rHlglyak",
    "xgboost_model.pkl": "https://drive.google.com/uc?id=1XI-yQ-wu8CSgZhQM91GgH3Ny3XdP0Ixz",
}
TARGET_DIR = "resources"

@st.cache_resource
def ensure_all_files():
    """Check and download missing files if necessary"""
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    missing_files = []
    for filename, url in FILES.items():
        path = os.path.join(TARGET_DIR, filename)
        if not os.path.exists(path):
            st.write(f"⬇️ Downloading missing file: {filename}")
            gdown.download(url, path, quiet=False)
            missing_files.append(filename)

    return TARGET_DIR

download_dir = ensure_all_files()

# ---------------------- #
# 3. Load Data and Models
# ---------------------- #
df = pd.read_csv(os.path.join(download_dir, "10_%_raw_data.csv"), low_memory=False).rename(
    columns={"reviewerID": "user_id", "asin": "item_id", "reviewText": "review_text"}
)

svd_model_path = os.path.join(download_dir, "trained_svd_model.pkl")
bert_vectors = np.load(os.path.join(download_dir, "bert_embeddings.npy"))
bert_asins = pd.read_csv(os.path.join(download_dir, "bert_asins.csv"))["asin"].tolist()
bert_item_id_to_idx = {asin: i for i, asin in enumerate(bert_asins)}

with open(os.path.join(download_dir, "xgboost_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)

# ---------------------- #
# 4. UI Setup
# ---------------------- #
st.sidebar.header("🔎 Filter Options")
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5)
category_options = df["category"].dropna().unique() if "category" in df.columns else []
selected_category = (
    st.sidebar.selectbox("Category", ["All"] + sorted(category_options.tolist()))
    if len(category_options) > 0
    else "All"
)

user_id = st.text_input("Enter User ID: (For example, 'AAP7PPBU72QFM')")
model_choice = st.selectbox("Choose Model", ["Hybrid", "SVD", "BERT", "Sentiment", "XGBoost", "NCF"])

# ---------------------- #
# 5. Generate Recommendations
# ---------------------- #
if st.button("Get Recommendations"):
    # --- SVD Predictions ---
    svd_recommended_items_df = get_svd_predictions_for_user_history(df, user_id, svd_model_path)

    # --- Sentiment Analysis ---
    sentiment_df = calculate_sentiment_for_items(df, svd_recommended_items_df)

    # --- BERT Similarity ---
    bert_similarity_df = calculate_bert_content_similarity(
        df, sentiment_df, user_id, bert_vectors, bert_item_id_to_idx
    )
    st.write(bert_similarity_df.head())

    # --- XGBoost Predictions ---
    xgb_predictions_df = calculate_xgboost(bert_similarity_df, xgb_model)

    # --- NCF Predictions ---
    ncf_predictions_df = get_ncf_predictions(
        xgb_predictions_df,
        os.path.join(download_dir, "ncf_model.pt"),
        os.path.join(download_dir, "user_item_mappings.pkl"),
        embedding_dim=64,
    )
    st.write(ncf_predictions_df.head())
    
    
# FILES = {
#     "10_%_raw_data.csv": "https://drive.google.com/uc?id=11VltwluUJR87OvO9v-E6ZRebMTNTq-ge",
#     "bert_asins.csv": "https://drive.google.com/uc?id=1hZNdi7EjCyQsdo5USDyMxdSNWQbW5gfb",
#     "bert_embeddings.npy": "https://drive.google.com/uc?id=1l5_VKscgAulltGSgZqbwFjqNhB86g-KH",
#     "ncf_model.pt": "https://drive.google.com/uc?id=1rDBZp7t-XsJlpp6Kj4upuO2RAkMxaxql",
#     "trained_svd_model.pkl": "https://drive.google.com/uc?id=1qoL7nBaIQqTimBZStVld8rCr0oekqd9m",
#     "user_item_mappings.pkl": "https://drive.google.com/uc?id=1WipR3wt_XIwsydaSHvlGyEt_rHlglyak",
#     "xgboost_model.pkl": "https://drive.google.com/uc?id=1XI-yQ-wu8CSgZhQM91GgH3Ny3XdP0Ixz",
# }

# TARGET_DIR = "resources"

# @st.cache_resource
# def ensure_all_files():
#     if not os.path.exists(TARGET_DIR):
#         os.makedirs(TARGET_DIR)

#     missing_files = []
#     for filename, url in FILES.items():
#         path = os.path.join(TARGET_DIR, filename)
#         if not os.path.exists(path):
#             print(f"⬇️ Downloading missing file: {filename}")
#             gdown.download(url, path, quiet=False)
#             missing_files.append(filename)

#     if missing_files:
#         print(f"✅ Downloaded missing files: {missing_files}")
#     else:
#         print("✅ All files already exist.")

#     return TARGET_DIR

# download_dir = ensure_all_files()

# # ----------------------
# # Paths & Loaders
# # ----------------------

# # --- Load the Raw DataFrame ---
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # df = load_reviews_df(os.path.join(BASE_DIR, 'data/raw/Electronics_5.json.gz')) # Memory is not enough for full dataset
# df = pd.read_csv(os.path.join(BASE_DIR, 'resources/10_%_raw_data.csv'), low_memory=False).rename(
#     columns = {'reviewerID': 'user_id', 'asin': 'item_id', 'reviewText': 'review_text'})

# # ----------------------
# # UI Setup
# # ----------------------
# st.set_page_config(page_title="Amazon Recommender", page_icon="📚")

# st.title("📚 Amazon Product Recommender")
# st.markdown("Choose a user or product to get recommendations. This app uses your final hybrid model outputs.")

# # Sidebar filters
# st.sidebar.header("🔎 Filter Options")
# min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5)

# category_options = df['category'].dropna().unique() if 'category' in df.columns else []
# selected_category = st.sidebar.selectbox("Category", ['All'] + sorted(category_options.tolist())) if len(category_options) > 0 else 'All'

# # ----------------------
# # Load Resources
# # ----------------------
# svd_model_path = os.path.join(BASE_DIR, 'resources/trained_svd_model.pkl')
# bert_vectors = np.load(os.path.join(BASE_DIR, 'resources/bert_embeddings.npy'))
# bert_asins = pd.read_csv(os.path.join(BASE_DIR, 'resources/bert_asins.csv'))["asin"].tolist()
# bert_item_id_to_idx = {bert_asins[i]: i for i in range(len(bert_asins))}

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# with open(os.path.join(BASE_DIR, 'resources/xgboost_model.pkl'), 'rb') as f:
#     xgb_model = pickle.load(f)

# # ----------------------
# # Tab 1: User Recs
# # ----------------------
# user_id = st.text_input("Enter User ID: (For example, 'AAP7PPBU72QFM')")

# model_choice = st.selectbox("Choose Model", ["Hybrid", "SVD", "BERT", "Sentiment", "XGBoost", "NCF"])

# if st.button("Get Recommendations"):
    
#     # ----------------------
#     # Create the DataFrame for Recommendations
#     # ----------------------
    
#     # --- SVD Predictions ---
#     svd_recommended_items_df = get_svd_predictions_for_user_history(df, user_id, svd_model_path)
#     # st.write(svd_recommended_items_df.head())
    
#     # --- Sentiment ---
#     sentiment_df = calculate_sentiment_for_items(df, svd_recommended_items_df)
#     # st.write(sentiment_df.head())
    
#     # --- Bert ---
#     bert_similarity_df = calculate_bert_content_similarity(df, sentiment_df, user_id, bert_vectors, bert_item_id_to_idx)
#     st.write(bert_similarity_df.head())
    
#     # --- XGBoost ---
    
#     xgb_predictions_df = calculate_xgboost(bert_similarity_df, xgb_model)
#     # st.write(xgb_predictions_df.head())
    
#     # --- NCF ---
#     model_path = os.path.join(BASE_DIR, 'resources/ncf_model.pt')
#     mapping_path = os.path.join(BASE_DIR, 'resources/user_item_mappings.pkl')
#     ncf_predictions_df = get_ncf_predictions(xgb_predictions_df, model_path, mapping_path, embedding_dim=64)
#     st.write(ncf_predictions_df.head())
    
    # while True:
    #     if user_id == "":
    #         st.write("No user found in database.")
    #         break

    #     user_recs = df[df['user_id'] == user_id]
    #     user_recs = user_recs.rename(
    #         columns={"svd_rating": "SVD",
    #             "bert_similarity": "BERT",
    #             "sentiment_score": "Sentiment",
    #             "xgb_pred_score": "XGBoost",
    #             "ncf_score": "NCF"})
        
    #     user_recs['Hybrid'] = 0.3 * user_recs['XGBoost'] + 0.25 * user_recs['NCF'] + 0.15 * user_recs['BERT'] + 0.15 * user_recs['SVD'] + 0.15 * user_recs ['Sentiment']

    #     if model_choice in user_recs.columns:
    #         user_recs = user_recs[user_recs[model_choice] >= min_rating]
    #         top_recs = user_recs.sort_values(model_choice, ascending=False).head(5)
    #         if len(user_recs[model_choice]) == 0:
    #             st.write(f"No recommendations found with {model_choice} scores ≥ {min_rating}. Try lowering the minimum rating.")
    #             break

    #     st.subheader(f"Top 5 Recommendations for User {user_id}")
    #     st.markdown(f" **Min: {user_recs[model_choice].min():.5f}, Max: {user_recs[model_choice].max():.5f}**")
    #     for _, row in top_recs.iterrows():
    #         st.markdown(f"**Product ID: {row['asin']}**")
    #         # st.write(f"🐹 Selected {model_choice} Score: {row.get(model_choice, 0):.5f}")
    #         st.write(f"📊 Hybrid Score: {row.get('Hybrid', 0):.5f}")
    #         st.write(f"⭐ SVD: {row.get('SVD', 0):.5f}")
    #         st.write(f"💬 Sentiment: {row.get('Sentiment', 0):.5f}")
    #         st.write(f"👅 BERT: {row.get('BERT', 0):.5f}")
    #         st.write(f"🌲 XGBoost: {row.get('XGBoost', 0):.5f}")
    #         st.write(f"🧠 NCF: {row.get('NCF', 0):.5f}")
            
    #         st.write("---")

    #     # 🔍 Visualization
    #     st.subheader("Score Comparison")
    #     plt.figure(figsize=(10, 4))
    #     plt.bar(top_recs['asin'], top_recs[model_choice], color='skyblue')
    #     plt.title(f"{model_choice} Scores for Top 5 Products")
    #     plt.xlabel("Product ID")
    #     plt.ylabel("Score")
    #     st.pyplot(plt)

    #     break