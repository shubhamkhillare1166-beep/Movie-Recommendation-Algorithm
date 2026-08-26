# 🎬 Netflix Movie Recommendation System

A content-based movie recommendation engine built with Python, Machine Learning, and Streamlit.

## 🚀 Live Demo
*(Paste your Streamlit Cloud link here once deployed!)*

---

## 📌 Overview
This application recommends movies based on metadata similarity using textual features like genres, plot keywords, top cast members, and directors.

It processes the **TMDB 5000 Movie Dataset** to vectorize movie "text soups" and computes dynamic recommendations using Cosine Similarity.

---

## 🛠️ How It Works

1. **Data Cleaning & Feature Engineering:** Extracted relevant metadata (`genres`, `keywords`, `cast`, `crew`) using Python's `ast` module.
2. **Text Soup Construction:** Combined key metadata fields into a single unified string for each movie.
3. **TF-IDF Vectorization:** Converted text descriptions into numerical feature matrices using `TfidfVectorizer` (filtering English stop words).
4. **Similarity Engine:** Calculated multi-dimensional similarity scores using `cosine_similarity`.
5. **Interactive UI:** Built a user-friendly dashboard using Streamlit for quick, real-time movie queries.

---

## 💻 Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, AST
* **Machine Learning:** Scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
* **Web Framework:** Streamlit

---

## 🏃 Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME


libraries used in the project

1) pandas
2) scikit-learn
3) streamlit
