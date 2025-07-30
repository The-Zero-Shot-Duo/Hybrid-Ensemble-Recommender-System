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
short_description: A hybrid recommender system using SVD, BERT, sentiment analysis, XGBoost, and NCF.
---

# 📚 Hybrid Ensemble Recommender System

A **hybrid product recommendation system** built with multiple models including:

- SVD-based Collaborative Filtering
- BERT-based Content Similarity
- Sentiment Analysis
- XGBoost Ranking
- Neural Collaborative Filtering (NCF)

The project provides an **interactive Streamlit UI** where users can input a user ID to receive personalized recommendations.

---

## 👥 Authors

- **Zhixiang Peng** - [GitHub Profile](https://github.com/Amos-Peng-127)
- **Rain Lin** - [GitHub Profile](https://github.com/TINYRAINYLIN)

---

## ✨ Features

✅ **Hybrid recommendation** using multiple models  
✅ **SVD-based collaborative filtering**  
✅ **Content-based recommendation** using BERT embeddings  
✅ **Sentiment analysis** to refine recommendations  
✅ **XGBoost ranking model**  
✅ **Neural Collaborative Filtering (NCF)** for deep learning-based recommendations  
✅ **Interactive Streamlit UI** with filtering options

---

## 📂 Project Structure

```
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── notebooks/helper.py  # Helper functions for recommendation logic
├── resources/           # Model files and embeddings (auto-downloaded)
└── README.md            # Project documentation
```

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Model Files

The app **automatically downloads required files** (CSV, embeddings, and models) from Google Drive on the first launch.

---

## 🛠 Tech Stack

- **Streamlit** – UI Framework
- **Python** – Core language
- **Pandas / NumPy** – Data handling
- **Matplotlib** – Visualization
- **gdown** – Downloading models from Google Drive
- **XGBoost / PyTorch** – Machine Learning models

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🔗 Links

- 🔴 **Live Demo (Hugging Face Spaces)**: [Your Space Link Here]
- 🟡 **GitHub Repo**: [Your GitHub Link Here]
