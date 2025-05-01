# YouTube Cross-Genre Engagement Analysis

## Project Overview
We set out to understand how users whose primary interest is **Entertainment** explore other YouTube categories. Using a 100 K–comment dataset (YT-100K) and supplemental video metadata, we built a pipeline for data cleaning, exploratory analysis, feature engineering, and baseline classification models to predict cross-genre engagement.

## Data Acquisition
- **YT-100K comments**: user comments with `video_id`, `commenter_channel_id`, `comment`, `votes`, and `category` labels.  
- **YouTube Trending Stats**: Kaggle “US trending” dump—with daily view/like/dislike counts and `publishedAt` timestamps—but that proved to have no overlapping IDs with our comment set.  
- **YouTube Data API v3** (planned): to fetch up-to-date `viewCount`, `likeCount`, `commentCount`, and `publishedAt` for our videos.

## Data Cleaning & EDA
- **Duplicates & missing values**: dropped duplicate comments, coerced `votes` to integers, removed rows missing text.  
- **Type fixes & outlier capping**: clipped `votes` and `comment_length` at the 99th percentile.  
- **Feature engineering**:  
  - Per-user comment counts by video and channel  
  - Comment length, category occurrence, and category ratios  
  - **Planned**: recency-weighted engagement, sentiment scores, popularity percentiles, cross-genre transition counts, latent taste embeddings.  
- **Visualizations**:  
  - Bar plots of comment counts by category  
  - Heatmaps of feature–genre correlations  
  - User×genre co-occurrence matrices

## Modeling
- **Baseline**: logistic regression models per non-Entertainment genre, predicting whether an “Entertainment” user ever comments in genre G.  
- **Evaluation**: mean predicted probability on a stratified hold-out; ROC-AUC used to gauge separability.  
- **Findings**: raw video stats from trending dump did not join (zero overlap), so per-user video features were all zeros—model predictions remained near zero for every genre.

## Challenges & Next Steps
1. **Data alignment**: video IDs in the Kaggle dump didn’t match YT-100K; we must pull metadata via the YouTube API.  
2. **Richer features**: implement recency, sentiment, popularity percentiles, transition counts, and latent embeddings to capture true cross-genre signals.  
3. **Model refinement**: once real video-level stats are merged, retrain with these enhanced features; explore other classifiers (random forests, gradient boosting) and cross-validation.  
4. **Interpretability**: quantify which new features most drive cross-genre engagement, and prepare downstream visualizations for non-technical stakeholders.
