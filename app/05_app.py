import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# ----------------------
# Define paths
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWS_PATH = os.path.join(BASE_DIR, '../data/processed/03_df_with_sentiment_2.csv')
PRODUCTS_PATH = os.path.join(BASE_DIR, '../data/processed/04_df_grouped.csv')
SVD_MODEL_PATH = os.path.join(BASE_DIR, '../models/svd_model.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(BASE_DIR, '../models/tfidf_vectorizer.pkl')
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, '../models/tfidf_matrix.npz')

# ----------------------
# Loaders
# ----------------------
@st.cache_data
def load_reviews():
    df = pd.read_csv(REVIEWS_PATH)
    df.dropna(subset=['asin'], inplace=True)
    return df

@st.cache_data
def load_products():
    df = pd.read_csv(PRODUCTS_PATH)
    df.dropna(subset=['asin'], inplace=True)
    df = df.drop_duplicates(subset=['asin']).reset_index(drop=True)
    return df

@st.cache_resource
def load_svd_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_tfidf_resources(vectorizer_path, matrix_path):
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    matrix = sparse.load_npz(matrix_path)
    return vectorizer, matrix

# ----------------------
# Load resources
# ----------------------
try:
    df_reviews = load_reviews()
    df_products = load_products()
    svd_model = load_svd_model(SVD_MODEL_PATH)
    vectorizer, tfidf_matrix = load_tfidf_resources(TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH)

    st.set_page_config(page_title="Amazon Recommender", page_icon="📚")
    # Add filters to sidebar
    st.sidebar.header("🔎 Filter Options")
    min_rating = st.sidebar.slider("Minimum Avg Rating", 1.0, 5.0, 3.5)

    # Add category filter if column exists
    if 'category' in df_reviews.columns:
        category_options = df_reviews['category'].dropna().unique()
        selected_category = st.sidebar.selectbox("Product Category", ['All'] + sorted(category_options.tolist()))
    else:
        selected_category = 'All'

    st.title("📚 Amazon Product Recommender")
    st.markdown("This version uses separate datasets for SVD and TF-IDF.")

    option = st.radio("Choose Recommendation Type:", ("User-based Recommendations", "Similar Products by Content"))

    if option == "User-based Recommendations":
        user_id = st.text_input("Enter User ID:")
        if user_id:
            user_items = df_reviews[df_reviews['reviewerID'] == user_id]['asin'].unique()
            all_items = df_reviews['asin'].unique()
            unseen_items = list(set(all_items) - set(user_items))

            predictions = []
            for item in unseen_items:
                pred = svd_model.predict(user_id, item)
                predictions.append((item, pred.est))

            top_items = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

            st.subheader(f"Top 5 Recommendations for User: {user_id}")
            for item_id, rating in top_items:
                product_info = df_reviews[df_reviews['asin'] == item_id]
                if not product_info.empty:
                    avg_rating = product_info['overall'].mean()

                    # 🔹 Apply minimum rating filter
                    if avg_rating < min_rating:
                        continue

                    # 🔹 Apply category filter if selected
                    if selected_category != 'All' and 'category' in product_info.columns:
                        if selected_category not in product_info['category'].values:
                            continue

                    product_info = product_info.iloc[0]
                    st.markdown(f"**Product ID: {item_id}**")
                    st.write(f"Predicted Rating: {rating:.2f}")
                    st.write(f"Average Sentiment Score: {product_info.get('sentiment_score', 'N/A'):.2f}")
                    st.write(f"Average Rating: {product_info.get('overall', 'N/A'):.2f}")
                    st.write("---")


    elif option == "Similar Products by Content":
        product_id = st.text_input("Enter Product ID:")
        if product_id:
            if product_id in df_products['asin'].values:
                idx_list = df_products[df_products['asin'] == product_id].index.tolist()
                if idx_list:
                    idx = idx_list[0]
                    product_vector = tfidf_matrix[idx]
                    similarities = cosine_similarity(product_vector, tfidf_matrix)[0]
                    sorted_idx = np.argsort(-similarities)

                    st.subheader(f"Top 5 Similar Products to: {product_id}")
                    shown = 0
                    for i in sorted_idx:
                        if i == idx:
                            continue
                        if i >= tfidf_matrix.shape[0] or i >= len(df_products):
                            continue
                        item = df_products.iloc[i]
                        st.markdown(f"**Product ID: {item['asin']}**")
                        st.write(f"Similarity Score: {similarities[i]:.2f}")
                        st.write(f"Sentiment Score: {item.get('sentiment_score', 'N/A'):.2f}")
                        st.write("---")
                        shown += 1
                        if shown >= 5:
                            break
                else:
                    st.warning("Product ID index not found.")
            else:
                st.warning("Product ID not found in product table.")

except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
except Exception as e:
    import traceback
    st.error("An unexpected error occurred:")
    st.text(traceback.format_exc())
