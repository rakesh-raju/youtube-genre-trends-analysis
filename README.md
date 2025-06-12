# YouTube Cross-Genre Engagement Analysis

## Project Overview
We set out to understand how users whose primary interest is Entertainment explore other YouTube categories. Using a 100K-comment dataset (YT-100K) and supplemental metadata, we built a pipeline for data cleaning, exploratory analysis, feature engineering, and baseline classification models to predict cross-genre engagement.

## Data Acquisition
- **YT-100K Comments**: A dataset of user comments with fields: `video_id`, `comment_id`, `commenter_channel_id`, `comment`, `votes`, and existing `category` labels.
- **YouTube Trending Stats (US)**: A Kaggle “US trending” dump containing daily view/like/dislike counts and timestamps. Attempts to join on `video_id` showed no overlap with YT-100K, indicating the need to fetch metadata via YouTube Data API.
- **YouTube Data API v3 (Planned)**: To fetch up-to-date `viewCount`, `likeCount`, `commentCount`, `publishedAt`, and verify video categories for accurate features.

## Data Cleaning & EDA
- **Duplicates & Missing Values**: Dropped duplicate comments, removed rows missing text (`comment`), coerced `votes` to integer (clipped outliers).
- **Type Fixes & Outlier Capping**: Converted fields to proper types; clipped `votes` and `comment_length` at the 99th percentile to reduce skew.
- **Category Standardization**: Lowercased and trimmed category labels; mapped category IDs from API JSON to human-readable names.
- **Feature Engineering**:
  - **Per-User Engagement**: 
    - Average video statistics per user (`view_count`, `likes`, `dislikes`, `comment_count`) after merging metadata.
    - Number of unique genres commented on.
    - Genre ratios: proportion of comments each user made in each genre.
    - Genre diversity (entropy) of user’s comment distribution across genres.
    - Recency-weighted engagement (planned).
    - Sentiment scores on comments (planned).
    - Popularity percentiles of commented videos (planned).
    - Cross-genre transition counts: sequences of user commenting across genres (planned).
    - Latent taste embeddings via matrix factorization or language models on comments (planned).
- **Exploratory Analysis & Visualizations**:
  - Bar plots of comment counts by category.
  - Heatmaps of feature–genre correlations.
  - User×genre co-occurrence matrices.
  - Distribution plots (histograms, boxplots) for numeric features (e.g., comment length, votes).
  - Time-based plots if timestamp metadata becomes available.

## Modeling
- **Baseline Models**:
  - For users whose majority category is 'entertainment', we built separate Logistic Regression models (with scaling and hyperparameter tuning) to predict whether they comment in other genres.
  - Evaluation via stratified hold-out; metrics: ROC-AUC for imbalanced targets, mean predicted probability reported for insight into likelihood.
- **Cross-Validation & Hyperparameter Tuning**:
  - Used `GridSearchCV` with `StratifiedKFold` to tune Logistic Regression regularization strength (`C`) per genre when sample sizes permit.
  - Skipped genres with insufficient positive/negative samples to avoid invalid splits.
- **Findings**:
  - Without properly joined video-level metadata, per-user video stats were zero or missing, resulting in near-zero predictive signals across genres.
  - After injecting synthetic or placeholder mappings for demonstration, the strongest signals came from genre ratios and diversity features.
  - Hyperparameter tuning produced robust model parameters (e.g., small `C` for regularization) but limited by data imbalance.
- **Next Steps & Recommendations**:
  - **Data Alignment**: Acquire real video metadata via YouTube Data API to ensure meaningful per-video features (views, likes, recency).
  - **Richer Features**: Implement recency-weighted counts, comment sentiment, popularity percentiles, transition patterns, and latent embeddings from comment text to capture nuanced preferences.
  - **Alternative Models**: Explore tree-based classifiers (Random Forest, Gradient Boosting) and ensemble methods to capture nonlinear relationships. Use cross-validation robustly.
  - **Address Class Imbalance**: Use techniques such as SMOTE, class weighting, or threshold tuning where positive samples are scarce.
  - **Interpretability**: Once meaningful features are available, analyze feature importances and coefficients to inform content strategy. Prepare visual summaries for stakeholders.
  - **Temporal Analysis**: Incorporate timestamp data to study how user preferences evolve over time.
  - **Scalability & Automation**: Automate data ingestion from YouTube API, schedule periodic updates, and pipeline retraining for ongoing insights.
- **Evaluation Metric Rationale**:
  - **ROC-AUC**: Suitable for imbalanced binary classification of whether an entertainment-majority user comments in another genre. Reflects separability independent of threshold.
  - **Mean Predicted Probability**: Provides interpretable likelihood insights for stakeholder discussions.
  - **Alternate Metrics (Planned)**: Precision-Recall AUC for highly imbalanced targets; use F1 or accuracy when classes balance improves.

## Project Organization
- **Repository Structure**:
  - `README.md`: Summary of objectives, methods, findings, next steps.
  - `notebooks/`: Jupyter notebooks with EDA, feature engineering, modeling sections. Headings and markdown cells clearly describe each step.
  - `data/`: Raw and cleaned datasets, with clear naming (`YT-100K_cleaned.csv`, metadata files).
  - `scripts/`: Python scripts for data pipeline (cleaning, feature generation, modeling).
  - `visualizations/`: Saved plot images (PNG) with readable labels and titles.
  - `.gitignore`: Exclude large raw data and API keys; include processed artifacts as needed.
- **Coding Best Practices**:
  - Import libraries with standard aliases (`pd`, `np`, `sns`, `plt`).
  - Functions and pipelines modularized; comments explain logic and assumptions.
  - Validation: ensure no errors, avoid long console outputs; use logging where appropriate.
  - Visualization: human-readable labels, descriptive titles, legible axes, appropriate scaling and subplots.
  - DataFrame operations: use pandas idioms for grouping, merging, handling missing values.

## Usage
1. **Clone repository** and set up environment (e.g., `requirements.txt` with `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, etc.).
2. **Place data files** in `data/` or configure API credentials for YouTube Data API.
3. **Run notebooks** sequentially: data cleaning & EDA → feature engineering → baseline modeling → advanced modeling.
4. **Review visualizations** in `visualizations/` and summary in `README.md`.
5. **Extend features** and retrain models with real metadata.

## Conclusion
This project lays the groundwork for understanding cross-genre engagement on YouTube. While initial models are limited by missing metadata, the architecture supports scalable feature engineering and modeling. Implementing richer data sources and advanced algorithms will enable robust insights for content creators and platform strategists. Future work should focus on API-based metadata ingestion, temporal dynamics, and deeper user preference embeddings to capture true cross-genre behaviors.

---

