---
title: Hybrid Ensemble Recommender System
emoji: 📚
colorFrom: pink
colorTo: green
sdk: streamlit
sdk_version: 1.33.0
app_file: app.py
pinned: false
license: mit
short_description: Hybrid recommender using SVD, BERT, Sentiment, XGBoost, NCF.
---

# 📚 Hybrid Ensemble Recommender System

A **Streamlit-based Amazon product recommender system** that combines multiple recommendation algorithms to generate **hybrid recommendations**.  
It integrates **SVD, BERT content similarity, sentiment analysis, XGBoost, and Neural Collaborative Filtering (NCF)**.

## 👥 Authors

- **Zhixiang Peng** - [GitHub Profile](https://github.com/Amos-Peng-127)
- **Rain Lin** - [GitHub Profile](https://github.com/TINYRAINYLIN)

## 🚀 Features

- **Hybrid recommendation** using multiple models
- Supports **SVD-based collaborative filtering**
- **Content-based recommendation** using BERT embeddings
- **Sentiment analysis** to refine recommendations
- **XGBoost ranking model**
- **Neural Collaborative Filtering (NCF)** for deep learning-based recommendations
- **Interactive Streamlit UI** with filtering options

## 📂 Project Structure

├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── notebooks/helper.py # Helper functions for recommendation logic
├── resources/ # Model files and embeddings
└── README.md # Project documentation

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install -r requirements.txt
```

▶️ Usage
Run the Streamlit app:
streamlit run app.py
Then open http://localhost:8501 in your browser.

📦 Model Files
The app automatically downloads required files (CSV, embeddings, models) from Google Drive when first launched.

🛠 Tech Stack
Streamlit – UI Framework

Python – Core language

Pandas / NumPy – Data handling

Matplotlib – Visualization

gdown – Downloading models from Google Drive

XGBoost / PyTorch – ML models

📜 License
This project is licensed under the MIT License.

🔗 Live Demo (Hugging Face Spaces): Your Space Link Here
📂 GitHub Repo: Your GitHub Link Here
