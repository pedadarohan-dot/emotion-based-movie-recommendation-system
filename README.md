# CineMatch 🎬

> A hybrid movie recommendation engine that understands not just what you watch — but how you feel.

CineMatch combines **content-based filtering**, **collaborative filtering via matrix factorization**, and **real-time emotion detection** from natural language to recommend movies that match both your taste *and* your current mood. Built on the MovieLens 32M dataset. Powered by a fine-tuned DistilRoBERTa emotion classifier.

---

## Why This Project

Most recommendation systems answer: *"What movies are like the ones you liked?"*

CineMatch answers: *"What movies should you watch **right now**, given what you like **and** how you feel?"*

If you're sad and you just watched *Interstellar*, a pure content-based system might recommend *Gravity* or *Arrival* — technically similar, emotionally wrong. CineMatch detects that you're sad and gently steers the recommendations toward *Drama*, *Romance*, and *Comedy* — movies that comfort, not overwhelm.

---

## Architecture

```
User Input: Movie Title + Mood Description
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
 [1] Content-Based          [2] Collaborative
     Filtering                   Filtering
  TF-IDF on Genres         Truncated SVD (k=50)
  + Cosine Similarity       on User-Movie Matrix
        │                        │
        │   Genre Score (0–1)    │   Collab Score (0–1)
        └───────────┬────────────┘
                    │
             [3] Emotion Layer
          DistilRoBERTa Classifier
          → Predicted Emotion
          → Genre Boost Mapping
                    │
                    ▼
         Final Score per Movie
         0.40 × Genre
       + 0.40 × Collaborative
       + 0.20 × Mood Boost
                    │
                    ▼
          Top 10 Recommendations
            (with score breakdown)
```

---

## System Design

### 1. Content-Based Filtering

Each movie's genre string (e.g., `Action|Sci-Fi|Thriller`) is tokenized and vectorized using **TF-IDF**:

```python
tfidf = TfidfVectorizer()
genre_embed = tfidf.fit_transform(df_movies["genres"])
```

Similarity between movies is computed using **cosine similarity** on these genre vectors. Given a seed movie, the top 20 genre-similar candidates are retrieved in O(n) time — no precomputed pairwise matrix (which would be 87K × 87K and infeasible).

### 2. Collaborative Filtering — Truncated SVD

Raw user-movie interaction data is dense: 32M ratings, 200K+ users, 87K+ movies. To handle this at scale:

- **Active user filter**: only users with ≥ 100 ratings (meaningful signal, not noise)
- **Popular movie filter**: only movies with ≥ 50 ratings (enough collaborative evidence)
- **Sampling**: top 10,000 active users for the interaction matrix

The User-Movie matrix is transposed to a **Movie-User matrix** and decomposed using Truncated SVD:

```python
svd = TruncatedSVD(n_components=50, random_state=42)
movie_embeddings = svd.fit_transform(movie_user_matrix)
```

Each movie is now a dense **50-dimensional latent vector** instead of a 10,000-dimensional sparse one. These latent dimensions capture hidden patterns — action preference, nostalgia, pacing, etc. — learned automatically from how 10,000 users collectively rated films.

Recommendation is then cosine similarity over these embeddings.

### 3. Emotion Layer — DistilRoBERTa

The emotion classifier is [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base), a fine-tuned DistilRoBERTa model that predicts 7 emotion classes from free-form text:

| Emotion  | Emoji | Genre Boost Strategy |
|----------|-------|----------------------|
| joy      | 😄    | Comedy · Adventure · Animation · Sci-Fi |
| sadness  | 😢    | Drama · Romance · Comedy |
| fear     | 😨    | Comedy · Family · Animation · Drama |
| anger    | 😠    | Action · Adventure · Comedy · Romance |
| surprise | 😲    | Sci-Fi · Mystery · Thriller |
| love     | ❤️    | Romance · Drama |
| neutral  | 😐    | Adventure · Comedy · Action |
| disgust  | 🤢    | Comedy · Drama · Sci-Fi |

The classifier runs on the user's natural language mood description:

```python
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_K=None
)
result = emotion_classifier("I feel restless and a bit anxious tonight")
# → [{'label': 'fear', 'score': 0.83}, ...]
```

The predicted emotion maps to a genre priority list. Each candidate movie receives a **mood score** based on genre overlap:

```python
matches = emotion_genres_set.intersection(movie_genres_set)
mood_score = len(matches) / len(emotion_genres_set)
```

### 4. Hybrid Scoring

```python
final_score = 0.40 * genre_score + 0.40 * collab_score + 0.20 * mood_score
```

Genre and collaborative signals carry equal weight (proven effective in the Netflix Prize literature). The mood layer is intentionally kept at 20% — enough to shift recommendations meaningfully without overriding clear user preferences.

---

## Dataset

[MovieLens 32M](https://grouplens.org/datasets/movielens/) — the full production-scale dataset from GroupLens.

| File | Rows | Description |
|------|------|-------------|
| `movies.csv` | 87,585 | movieId, title, genres |
| `ratings.csv` | 32,000,204 | userId, movieId, rating, timestamp |

After filtering (active users ≥ 100 ratings, popular movies ≥ 50 ratings, top 10K users):

| Metric | Value |
|--------|-------|
| Users in matrix | 10,000 |
| Movies in matrix | ~20,000 |
| SVD latent factors | 50 |
| Sparsity | ~97% |

---

## Example

**Input:**
- Favourite Movie: `Interstellar`
- How you're feeling: `I'm feeling a bit sad and nostalgic tonight`

**Emotion detected:** 😢 Sadness → boosting Drama · Romance · Comedy

**Output (top 5):**

| Rank | Movie | Genre Score | Collab Score | Mood Score | Final |
|------|-------|-------------|--------------|------------|-------|
| 🥇 | Gravity (2013) | 0.91 | 0.87 | 0.50 | 0.81 |
| 🥈 | The Martian (2015) | 0.89 | 0.85 | 0.25 | 0.74 |
| 🥉 | Contact (1997) | 0.88 | 0.81 | 0.50 | 0.74 |
| 4 | Her (2013) | 0.72 | 0.79 | 0.75 | 0.76 |
| 5 | Arrival (2016) | 0.85 | 0.84 | 0.25 | 0.73 |

*Her* gets mood-boosted because it's Romance/Drama and aligns with a sad, nostalgic state — even though it ranks lower on pure content/collab signals.

---

## UI — Gradio

The app ships with a fully styled **Gradio Blocks** interface:

- Dark cinema-themed layout (CSS variables, custom font via Google Fonts)
- Emotion pill with dynamic color per detected mood
- Per-movie score breakdown with mini progress bars (Genre · Collab · Mood)
- Animated rank badges (Gold / Silver / Bronze / #N)
- SVG donut chart per movie showing overall score
- Quick-start examples pre-loaded
- Fully responsive (collapses to single column on mobile)

```
┌──────────────────────────────────────────────────────────┐
│  CineMatch                            🟢 Live · All loaded│
│  Hybrid AI recommendation engine                         │
│  🧮 TF-IDF  🔢 SVD·50  🧠 DistilRoBERTa  🎬 ML-32M      │
├───────────────────┬──────────────────────────────────────┤
│  Your Preferences │  Your Recommendations                │
│                   │                                      │
│  Favourite Movie  │  🎬 Based on Interstellar            │
│  ┌─────────────┐  │  😢 Sadness · boosting Drama · ...   │
│  │ Interstellar│  │  ⚗️  40% Genre · 40% Collab · 20%   │
│  └─────────────┘  │                                      │
│                   │  #1 GOLD  Gravity          [0.81]    │
│  How are you      │  ████ Genre  ████ Collab  ██ Mood    │
│  feeling?         │                                      │
│  ┌─────────────┐  │  #2 SILVER The Martian     [0.74]    │
│  │ sad tonight │  │  ...                                 │
│  └─────────────┘  │                                      │
│                   │                                      │
│  [✦ Get Recs]     │                                      │
└───────────────────┴──────────────────────────────────────┘
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/pedadarohan-dot/emotion-based-movie-recommendation-system.git
cd emotion-based-movie-recommendation-system

# 2. Install dependencies
pip install pandas numpy scikit-learn transformers gradio torch

# 3. Download MovieLens 32M
#    → https://grouplens.org/datasets/movielens/32m/
#    Extract so the structure is:
#    emotion-based-movie-recommendation-system/
#    └── ml-32m/
#        ├── movies.csv
#        └── ratings.csv

# 4. Run
python app.py
# → Open http://127.0.0.1:7865
```

> ⚠️ **First run:** The SVD computation over 10K users takes ~2–5 minutes depending on hardware. Subsequent runs with cached embeddings will be faster. The emotion model (~330MB) downloads automatically from HuggingFace on first launch.

---

## Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
transformers
torch
gradio >= 4.0
```

No GPU required. Runs on CPU. Tested on macOS and Linux.

---

## Project Evolution

This project was built incrementally — no grand design up front, just solving one problem at a time.

```
Stage 1: Content-Based Only
   TF-IDF on genres + cosine similarity
   → Works, but ignores user behavior entirely

Stage 2: Add Collaborative Filtering
   User-Movie matrix + cosine similarity on raw rows
   → Memory crash: 87K × 87K similarity matrix is infeasible

Stage 3: Engineer Around Scale
   Compute similarity on-the-fly (input movie vs all)
   → Solved memory, but still sparse high-dim vectors

Stage 4: Matrix Factorization
   Truncated SVD → 50 latent factors per movie
   → Compact, dense, meaningful representations

Stage 5: Hybrid Scoring
   Combine genre + collab with weighted average
   → Better than either alone

Stage 6: Emotion Layer
   DistilRoBERTa classifier → genre boost mapping
   → Recommendations that adapt to the moment
```

The biggest lesson: **a correct solution is not always a practical solution.** The moment you hit a real dataset, you're forced to think about memory, sparsity, and approximation — not just math.

---

## What's Next

- [ ] Matrix Factorization with **ALS** (Alternating Least Squares) for implicit feedback
- [ ] User profile persistence — build a taste model over sessions
- [ ] Emotion confidence threshold — fall back to neutral if classifier is uncertain
- [ ] TMDB API integration — pull posters, ratings, and trailers into the UI
- [ ] Deploy on HuggingFace Spaces

---

## Acknowledgements

- [GroupLens Research](https://grouplens.org/) for the MovieLens 32M dataset
- [Jochen Hartmann](https://huggingface.co/j-hartmann) for `emotion-english-distilroberta-base`
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Gradio](https://www.gradio.app/) for the UI framework

---

## License

MIT
