import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from transformers import pipeline
import gradio as gr
# Dataset Understanding
df_movies = pd.read_csv("ml-32m/movies.csv")
'''print(df_movies.head())
print(df_movies.columns)
print(df_movies.shape)
print(df_movies.size)
print(df_movies.info)
print(df_movies["genres"].head())'''

# Replacing | in the genres column of the movies dataset
df_movies["genres"] = df_movies["genres"].str.replace("|", " ")
# print(df_movies["genres"])

# Using Term Frequency Inverse Document Frequency - It makes all the text based content and generates it in numbers. It is a combination of frequency of the word and the inverse frequency of the word.
tfidf = TfidfVectorizer()
genre_embed = tfidf.fit_transform(df_movies["genres"])
'''print("Genre Embed Shape: ")
print(genre_embed.shape)
print(tfidf.get_feature_names_out())'''''  # It lists out all the features in the tfidf

# Now I will use cosine similarity to find the similarity between the movies according to genres
user = input("Enter your favourite movie: ")
matching_rows = df_movies[df_movies["title"].str.contains(user, case=False, na=False)]
movie_id = matching_rows["movieId"]             # It takes the movieId of the user input movie
movie_index = matching_rows.index[0]            # It takes the movie index of the user input movie
movie_genre_vector = genre_embed[movie_index]
similarity_genre_score = cosine_similarity(movie_genre_vector,
                                     genre_embed)

# Similarity movies
similar_genre_movies = list(enumerate(similarity_genre_score[0]))
sorted_genre_movies = sorted(
    similar_genre_movies,
    key=lambda x: x[1],                            # This is arranging it in descending order according to the similarity score
    reverse=True
)

# Checking the ratings to make a hybrid recommendation system
df_ratings = pd.read_csv("ml-32m/ratings.csv")
'''print(df_ratings.head())
print(df_ratings.shape)
print(df_ratings["userId"].nunique())               # This is used to check all the unique user
print(df_ratings["movieId"].nunique())              # This is used to check all the unique movies
print(df_ratings["rating"].describe())              # This describes all about the ratings
print(df_ratings["rating"].value_counts().sort_index())'''         # This represents the count about the ratings
user_counts = (df_ratings["userId"].value_counts())
# print(user_counts.describe())

# For checking the perfect threshold
'''for threshold in [50, 100, 200]:
    print(
        threshold,
        (user_counts >= threshold).sum()
    )'''

# Filtering the active users
active_users = user_counts[user_counts >= 100].index                # It takes all the active users who have rated for above 100 movies
sample_users = active_users[:10000]                                 # It takes all the first 10000 users
df_ratings_filtered = df_ratings[df_ratings["userId"].isin(sample_users)]
# print(df_ratings_filtered)

# Filter for popular movies
movie_counts = df_ratings["movieId"].value_counts()

# Finding out the threshold
'''for threshold in [20, 50, 100]:
    print(
        threshold,
        (movie_counts >= threshold).sum()
    )'''

popular_movies = movie_counts[movie_counts >= 50].index
df_movies_filtered = df_ratings[df_ratings["movieId"].isin(popular_movies)]
# print(df_movies_filtered)

# User-Movie matrix
user_movie_matrix = df_ratings_filtered.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

movie_user_matrix = user_movie_matrix.T
movie_user_matrix = movie_user_matrix.fillna(0)

# movie_ratings_vector = movie_user_matrix.loc[movie_id]
# similarity_rating_score = cosine_similarity(movie_ratings_vector, movie_user_matrix)

# SVD embedding = It divides the genres into factors and it is easy to deal with NaN instead of 0 in hybrid recommendation system
svd = TruncatedSVD(
    n_components=50,
    random_state=42
)
movie_embeddings = svd.fit_transform(movie_user_matrix)             # It is applied to the movie_user_matrix. First before applying it we compare all 10000 users ratings for each movie. Now it is truncated to 50 latent factors
movie_vector = movie_embeddings[movie_index]
similarity_rating_score = cosine_similarity(movie_vector.reshape(1, -1), movie_embeddings)

# Similar rating movies
similar_rating_movies = list(enumerate(similarity_rating_score[0]))
sorted_rating_movies = sorted(
    similar_rating_movies,
    key=lambda x:x[1],
    reverse=True
)

# Hybrid Recommendation system: Uses both genres and ratings

genres = {}
ratings = {}
for movie in sorted_genre_movies[1:21]:
    index = movie[0]
    score = movie[1]
    movie_name = df_movies.iloc[index]["title"]
    genres[movie_name] = score

for movie in sorted_rating_movies[1:21]:
    index = movie[0]
    score = movie[1]
    movie_name = df_movies.iloc[index]["title"]
    ratings[movie_name] = score

hybrid_scores = {}
all_movies = set(genres.keys()) | set(ratings.keys())
for movie in all_movies:
    genre_score = genres.get(movie, 0)
    ratings_score = ratings.get(movie, 0)
    hybrid_score = (0.5 * genre_score + 0.5 * ratings_score)
    hybrid_scores[movie] = hybrid_score

sorted_hybrid_score = sorted(
    hybrid_scores.items(),
    key=lambda x:x[1],
    reverse=True
)

'''for movie, score in sorted_hybrid_score[:10]:
    print(movie)'''

# First we are importing the pretrained emotion based transformer and the giving it language and then allowing it to predict my mood
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_K=None
)
user_mood = input("How are you feeling today bro?: ")
result = emotion_classifier(user_mood)
predicted_mood = result[0]["label"]
print(predicted_mood)

# We will do mapping so that we can boost the mood of the users.
emotion_to_genres = {
    "joy": ["Comedy", "Adventure", "Animation", "Sci-Fi"],

    "sadness": ["Drama", "Romance", "Comedy"],

    "fear": ["Comedy", "Family", "Animation", "Drama", "Romance"],

    "anger": ["Action", "Adventure", "Comedy", "Romance"],

    "surprise": ["Sci-Fi", "Mystery", "Thriller"],

    "love": ["Romance", "Drama"],

    "neutral": ["Adventure", "Comedy", "Action"]
}

emotion_genres = emotion_to_genres[predicted_mood]
emotion_genres_set = set(emotion_genres)                # Because we want only unique emotions
mood_scores = {}
for _, row in df_movies.iterrows():
    movie_title = row["title"]
    movie_genres = row["genres"]
    movie_genres_set = set(movie_genres.split("|"))
    matches = emotion_genres_set.intersection(movie_genres_set)
    mood_score = len(matches) / len(emotion_genres_set)
    mood_scores[movie_title] = mood_score

final_scores = {}
for movie in genres:
    genre_score = genres.get(movie, 0)
    svd_score = ratings.get(movie, 0)
    mood_score = mood_scores.get(movie, 0)
    final_score = (0.4*genre_score + 0.4*svd_score + 0.2*mood_score)
    final_scores[movie] = final_score

sorted_movies = sorted(
    final_scores.items(),
    key=lambda x:x[1],
    reverse=True
)
for movie in sorted_movies[:10]:
    print(movie[0])

# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE GRADIO UI/UX CODE FOR YOUR RECOMMENDATION APP
# ─────────────────────────────────────────────────────────────────────────────

# Fast lookups for SVD indexing to prevent crashes
movie_ids = list(movie_user_matrix.index)
movie_id_to_emb_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
movie_id_to_title = dict(zip(df_movies["movieId"], df_movies["title"]))
title_to_genres = dict(zip(df_movies["title"], df_movies["genres"]))

# Meta mapping for UI styling based on predicted emotion
EMOTION_META = {
    "joy": {"emoji": "😄", "color": "#FBBF24", "name": "Joy", "preview": "Comedy · Adventure · Animation · Sci-Fi"},
    "sadness": {"emoji": "😢", "color": "#60A5FA", "name": "Sadness", "preview": "Drama · Romance · Comedy"},
    "fear": {"emoji": "😨", "color": "#F87171", "name": "Fear",
             "preview": "Comedy · Family · Animation · Drama · Romance"},
    "anger": {"emoji": "😠", "color": "#EF4444", "name": "Anger", "preview": "Action · Adventure · Comedy · Romance"},
    "surprise": {"emoji": "😲", "color": "#A78BFA", "name": "Surprise", "preview": "Sci-Fi · Mystery · Thriller"},
    "love": {"emoji": "❤️", "color": "#F472B6", "name": "Love", "preview": "Romance · Drama"},
    "neutral": {"emoji": "😐", "color": "#9CA3AF", "name": "Neutral", "preview": "Adventure · Comedy · Action"},
    "disgust": {"emoji": "🤢", "color": "#34D399", "name": "Disgust", "preview": "Comedy · Drama · Sci-Fi"}
}


def get_recommendations(movie_title: str, mood_text: str) -> str:
    if not movie_title.strip():
        return _idle()

    # Search for matching movie
    matching_rows = df_movies[df_movies["title"].str.contains(movie_title, case=False, na=False, regex=False)]
    if matching_rows.empty:
        return _error(f"Could not find any movies containing '{movie_title}' in the database.")

    movie_row = matching_rows.iloc[0]
    found_title = movie_row["title"]
    movie_id = movie_row["movieId"]
    movie_index = matching_rows.index[0]

    # 1. Genre similarity
    movie_genre_vector = genre_embed[movie_index]
    similarity_genre_score = cosine_similarity(movie_genre_vector, genre_embed)[0]
    similar_genre_movies = list(enumerate(similarity_genre_score))
    sorted_genre_movies = sorted(similar_genre_movies, key=lambda x: x[1], reverse=True)

    genres_scores = {}
    for movie in sorted_genre_movies:
        idx = movie[0]
        score = movie[1]
        title = df_movies.iloc[idx]["title"]
        if title == found_title:
            continue
        genres_scores[title] = score
        if len(genres_scores) >= 20:
            break

    # 2. Collaborative filtering (SVD) similarity
    ratings_scores = {}
    if movie_id in movie_id_to_emb_idx:
        emb_idx = movie_id_to_emb_idx[movie_id]
        movie_vector = movie_embeddings[emb_idx]
        similarity_rating_score = cosine_similarity(movie_vector.reshape(1, -1), movie_embeddings)[0]
        similar_rating_movies = list(enumerate(similarity_rating_score))
        sorted_rating_movies = sorted(similar_rating_movies, key=lambda x: x[1], reverse=True)

        for movie in sorted_rating_movies:
            idx = movie[0]
            score = movie[1]
            mid = movie_ids[idx]
            title = movie_id_to_title.get(mid)
            if not title or title == found_title:
                continue
            ratings_scores[title] = score
            if len(ratings_scores) >= 20:
                break

    # 3. Emotion classification and mood boost
    predicted_mood = "neutral"
    if mood_text.strip():
        try:
            result = emotion_classifier(mood_text.strip())
            predicted_mood = result[0][0]["label"] if isinstance(result[0], list) else result[0]["label"]
        except Exception as e:
            print(f"Error classifying emotion: {e}")
            predicted_mood = "neutral"

    meta = EMOTION_META.get(predicted_mood, EMOTION_META["neutral"])
    emoji = meta["emoji"]
    color = meta["color"]
    mood_name = meta["name"]
    mood_genres_preview = meta["preview"]

    emotion_genres = emotion_to_genres.get(predicted_mood, ["Adventure", "Comedy", "Action"])
    emotion_genres_set = set(emotion_genres)

    # 4. Calculate final hybrid scores (Limit iteration strictly to candidates)
    candidates = set(genres_scores.keys()) | set(ratings_scores.keys())

    candidate_mood_scores = {}
    genres_str_map = {}

    for title in candidates:
        movie_genres = title_to_genres.get(title, "")
        genres_str_map[title] = movie_genres

        # Split by space (since | was replaced by space in your dataset)
        movie_genres_set = set(movie_genres.split())
        matches = emotion_genres_set.intersection(movie_genres_set)
        mood_score = len(matches) / len(emotion_genres_set) if emotion_genres_set else 0.0
        candidate_mood_scores[title] = mood_score

    # Compute final hybrid scores (40% genre, 40% collab, 20% mood)
    final_scores = {}
    for title in genres_scores:  # Limiting final recommendation candidates to the top genre matches as in your original formula
        g_score = genres_scores.get(title, 0.0)
        r_score = ratings_scores.get(title, 0.0)
        m_score = candidate_mood_scores.get(title, 0.0)

        final_score = 0.4 * g_score + 0.4 * r_score + 0.2 * m_score
        final_scores[title] = final_score

    sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top10 = sorted_candidates[:10]

    if not top10:
        return _error(f"No recommendations could be generated for '{movie_title}'.")

    return _render(
        top10, found_title,
        genres_scores, ratings_scores, candidate_mood_scores, genres_str_map,
        emoji, color, mood_name, mood_genres_preview
    )


# ── HTML RENDERERS ────────────────────────────────────────────────────────────

def _idle() -> str:
    return """
    <div class="state-box">
        <div class="state-icon">🎬</div>
        <p class="state-title">Your cinema awaits</p>
        <p class="state-sub">Enter a movie you love and describe your mood — we'll handle the rest.</p>
    </div>"""


def _error(msg: str) -> str:
    return f"""
    <div class="state-box error">
        <div class="state-icon">⚠️</div>
        <p class="state-title">Movie not found</p>
        <p class="state-sub">{msg}</p>
    </div>"""


def _render(top10, found_title, genres_dict, ratings_dict,
            mood_scores, genres_str_map,
            emoji, color, mood_name, mood_genres_preview) -> str:
    RANK_STYLES = [
        ("#FFD700", "GOLD"),
        ("#C0C0C0", "SILVER"),
        ("#CD7F32", "BRONZE"),
    ]

    cards = ""
    for rank, (title, score) in enumerate(top10, 1):
        g = genres_dict.get(title, 0.0)
        r = ratings_dict.get(title, 0.0)
        m = mood_scores.get(title, 0.0)

        raw_gen = genres_str_map.get(title, "")
        genre_tags = "".join(
            f'<span class="gtag">{g_name.strip()}</span>'
            for g_name in raw_gen.split(" ") if g_name.strip()
        )

        rc, rl = RANK_STYLES[rank - 1] if rank <= 3 else ("#4F46E5", f"#{rank}")
        pct_g = int(g * 100)
        pct_r = int(r * 100)
        pct_m = int(m * 100)
        circ_pct = int(score * 94.25)
        delay = (rank - 1) * 0.07

        cards += f"""
        <div class="mcard" style="animation-delay:{delay:.2f}s">
            <div class="mcard-left">
                <div class="rank-badge" style="color:{rc};border-color:{rc}33;background:{rc}11">{rl}</div>
                <div class="score-donut">
                    <svg viewBox="0 0 44 44">
                        <circle cx="22" cy="22" r="18" fill="none" stroke="#1a1a2e" stroke-width="4"/>
                        <circle cx="22" cy="22" r="18" fill="none" stroke="{rc}" stroke-width="4"
                            stroke-dasharray="{circ_pct} 113.1"
                            stroke-dashoffset="28.27" stroke-linecap="round"/>
                    </svg>
                    <span class="donut-val">{score:.2f}</span>
                </div>
            </div>
            <div class="mcard-body">
                <div class="mcard-title">{title}</div>
                <div class="genre-tags">{genre_tags}</div>
                <div class="mini-bars">
                    <div class="mbar-row">
                        <span class="mbar-lbl">Genre</span>
                        <div class="mbar-track"><div class="mbar-fill mbar-genre" style="width:{pct_g}%"></div></div>
                        <span class="mbar-num">{g:.2f}</span>
                    </div>
                    <div class="mbar-row">
                        <span class="mbar-lbl">Collab</span>
                        <div class="mbar-track"><div class="mbar-fill mbar-collab" style="width:{pct_r}%"></div></div>
                        <span class="mbar-num">{r:.2f}</span>
                    </div>
                    <div class="mbar-row">
                        <span class="mbar-lbl">Mood</span>
                        <div class="mbar-track"><div class="mbar-fill mbar-mood" style="width:{pct_m}%"></div></div>
                        <span class="mbar-num">{m:.2f}</span>
                    </div>
                </div>
            </div>
        </div>"""

    return f"""
    <div class="results-root">
        <div class="meta-strip">
            <div class="meta-pill film">
                <span class="pill-icon">🎬</span>
                <span class="pill-body">Based on <strong>{found_title}</strong></span>
            </div>
            <div class="meta-pill mood" style="color:{color};border-color:{color}55;background:{color}11">
                <span class="pill-icon">{emoji}</span>
                <span class="pill-body">
                    <strong>{mood_name}</strong>
                    <span class="pill-sub">boosting {mood_genres_preview}</span>
                </span>
            </div>
            <div class="meta-pill algo">
                <span class="pill-icon">⚗️</span>
                <span class="pill-body">40% Genre · 40% Collab · 20% Mood</span>
            </div>
        </div>
        <div class="legend-row">
            <span class="leg"><i class="ld ld-genre"></i>Genre similarity</span>
            <span class="leg"><i class="ld ld-collab"></i>Collaborative (SVD)</span>
            <span class="leg"><i class="ld ld-mood"></i>Mood boost</span>
        </div>
        <div class="card-list">{cards}</div>
    </div>"""


# ── CSS STYLING ───────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --ink:   #05050F;
    --surf:  #0C0C1D;
    --surf2: #111127;
    --surf3: #181832;
    --rim:   rgba(255,255,255,0.06);
    --rim2:  rgba(255,255,255,0.10);
    --neon:  #7B61FF;
    --neon2: #FF61DC;
    --gold:  #FFD166;
    --txt:   #E8E6FF;
    --txt2:  #7B78A8;
    --r:     16px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body,
.gradio-container {
    background: var(--ink) !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--txt) !important;
}
.gradio-container { max-width: 1300px !important; margin: 0 auto !important; padding: 0 !important; }
footer { display: none !important; }

/* HERO */
.hero {
    position: relative;
    text-align: center;
    padding: 56px 24px 44px;
    overflow: hidden;
    border-bottom: 1px solid var(--rim2);
}

.hero::before,
.hero::after {
    content: '';
    position: absolute;
    border-radius: 50%;
    filter: blur(90px);
    opacity: .35;
    animation: drift 12s ease-in-out infinite alternate;
}
.hero::before {
    width: 520px; height: 520px;
    background: radial-gradient(circle, #7B61FF 0%, transparent 70%);
    top: -200px; left: -100px;
}
.hero::after {
    width: 420px; height: 420px;
    background: radial-gradient(circle, #FF61DC 0%, transparent 70%);
    top: -160px; right: -80px;
    animation-delay: -6s;
}
@keyframes drift {
    from { transform: translate(0,0) scale(1); }
    to   { transform: translate(30px,20px) scale(1.08); }
}

.hero-grid {
    position: absolute;
    inset: 0;
    background-image: radial-gradient(circle, rgba(255,255,255,.05) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
}

.hero-eyebrow {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(123,97,255,.12);
    border: 1px solid rgba(123,97,255,.25);
    border-radius: 999px;
    padding: 5px 16px;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #A89EFF;
    margin-bottom: 18px;
}
.pulse {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #34D399;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.8)} }

.hero h1 {
    position: relative;
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(4rem, 9vw, 7rem);
    letter-spacing: .04em;
    line-height: .95;
    background: linear-gradient(135deg, #fff 0%, #C8BEFF 35%, #FF9BF5 70%, #FFD166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}
.hero-sub {
    position: relative;
    color: var(--txt2);
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: .02em;
    max-width: 560px;
    margin: 0 auto 24px;
}
.tech-pills {
    position: relative;
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
}
.tpill {
    background: var(--surf2);
    border: 1px solid var(--rim2);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: .73rem;
    color: var(--txt2);
    font-weight: 500;
}

/* BODY GRID */
.body-wrap {
    display: grid;
    grid-template-columns: 360px 1fr;
    min-height: calc(100vh - 280px);
}
@media(max-width:860px) { .body-wrap { grid-template-columns: 1fr; } }

/* LEFT PANEL */
.left-panel {
    background: var(--surf);
    border-right: 1px solid var(--rim2);
    padding: 32px 26px 40px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}
.sect-label {
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--txt2);
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.sect-label::after { content:''; flex:1; height:1px; background:var(--rim2); }

.block { background: transparent !important; border: none !important; padding: 0 !important; box-shadow: none !important; }

label > span:first-child {
    font-family: 'Outfit', sans-serif !important;
    font-size: .67rem !important;
    font-weight: 700 !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    color: #6B6894 !important;
}
textarea, input[type="text"] {
    background: var(--surf2) !important;
    border: 1px solid var(--rim2) !important;
    border-radius: 12px !important;
    color: var(--txt) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: .95rem !important;
    font-weight: 400 !important;
    padding: 13px 16px !important;
    transition: border-color .2s, box-shadow .2s !important;
    resize: none !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--neon) !important;
    box-shadow: 0 0 0 3px rgba(123,97,255,.18) !important;
    outline: none !important;
}
textarea::placeholder, input::placeholder { color: #35334F !important; }

button.lg.primary {
    background: linear-gradient(135deg, #7B61FF 0%, #FF61DC 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.05rem !important;
    letter-spacing: .1em !important;
    padding: 16px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: transform .15s, box-shadow .2s !important;
}
button.lg.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 36px rgba(123,97,255,.50) !important;
}
button.lg.primary:active { transform: translateY(0) !important; }

.weight-card {
    background: var(--surf2);
    border: 1px solid var(--rim2);
    border-radius: 14px;
    padding: 16px 18px;
}
.wrow {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--rim);
    font-size: .82rem;
}
.wrow:last-child { border-bottom: none; }
.wrow-label { color: var(--txt2); display: flex; align-items: center; gap: 8px; }
.wrow-val   { font-weight: 700; font-size: .88rem; }

table.gr-samples-table {
    background: transparent !important;
    border-collapse: separate !important;
    border-spacing: 0 4px !important;
    width: 100% !important;
}
table.gr-samples-table thead { display: none !important; }
table.gr-samples-table td {
    background: var(--surf2) !important;
    border: 1px solid var(--rim) !important;
    border-radius: 9px !important;
    color: var(--txt2) !important;
    font-size: .8rem !important;
    padding: 8px 12px !important;
    cursor: pointer !important;
    transition: background .15s, color .15s, border-color .15s !important;
    font-family: 'Outfit', sans-serif !important;
}
table.gr-samples-table td:hover {
    background: rgba(123,97,255,.15) !important;
    color: var(--txt) !important;
    border-color: rgba(123,97,255,.35) !important;
}

/* RIGHT PANEL */
.right-panel {
    background: var(--ink);
    padding: 32px 28px;
    overflow-y: auto;
}

/* IDLE / ERROR STATES */
.state-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 100px 24px;
    text-align: center;
    color: var(--txt2);
}
.state-box.error { color: #F87171; }
.state-icon  { font-size: 3.6rem; margin-bottom: 18px; }
.state-title { font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem; letter-spacing: .06em; color: var(--txt); margin-bottom: 8px; }
.state-sub   { font-size: .9rem; line-height: 1.7; max-width: 340px; }

/* RESULTS */
.results-root { animation: fadeUp .4s ease both; }
@keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }

.meta-strip { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.meta-pill {
    display: inline-flex;
    align-items: flex-start;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 12px;
    border: 1px solid;
    font-size: .8rem;
    line-height: 1.5;
}
.meta-pill.film { background: rgba(123,97,255,.08); border-color: rgba(123,97,255,.22); color: #C4BAFF; }
.meta-pill.film strong { color: #fff; }
.meta-pill.algo { background: rgba(255,209,102,.05); border-color: rgba(255,209,102,.15); color: #78716C; font-size: .74rem; }
.pill-icon { font-size: 1rem; line-height: 1.5; }
.pill-sub  { display: block; font-size: .7rem; opacity: .6; margin-top: 1px; }

.legend-row { display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 18px; font-size: .73rem; font-weight: 500; color: var(--txt2); }
.leg { display: flex; align-items: center; gap: 6px; }
.ld  { display: inline-block; width: 22px; height: 5px; border-radius: 3px; }
.ld-genre  { background: linear-gradient(90deg,#7B61FF,#A78BFA); }
.ld-collab { background: linear-gradient(90deg,#0EA5E9,#38BDF8); }
.ld-mood   { background: linear-gradient(90deg,#FFD166,#FBBF24); }

/* MOVIE CARDS */
.card-list { display: flex; flex-direction: column; gap: 10px; }

.mcard {
    display: grid;
    grid-template-columns: 90px 1fr;
    gap: 18px;
    background: var(--surf);
    border: 1px solid var(--rim2);
    border-radius: var(--r);
    padding: 18px 20px;
    transition: border-color .2s, background .2s, transform .15s;
    animation: slideIn .35s ease both;
    position: relative;
    overflow: hidden;
}
.mcard::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(123,97,255,.03) 0%, transparent 60%);
    pointer-events: none;
}
@keyframes slideIn { from{opacity:0;transform:translateX(-12px)} to{opacity:1;transform:translateX(0)} }
.mcard:hover {
    border-color: rgba(123,97,255,.35);
    background: var(--surf2);
    transform: translateX(4px);
}

.mcard-left { display: flex; flex-direction: column; align-items: center; gap: 10px; }

.rank-badge {
    font-family: 'Bebas Neue', sans-serif;
    font-size: .85rem;
    letter-spacing: .1em;
    border: 1px solid;
    border-radius: 6px;
    padding: 2px 8px;
    text-align: center;
}

.score-donut { position: relative; width: 56px; height: 56px; }
.score-donut svg { width:100%; height:100%; transform:rotate(-90deg); }
.donut-val {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: .72rem;
    font-weight: 700;
    color: var(--txt);
}

.mcard-body { min-width: 0; }

.mcard-title {
    font-weight: 700;
    font-size: 1rem;
    color: var(--txt);
    margin-bottom: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.genre-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; }
.gtag {
    font-size: .63rem;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
    background: var(--surf3);
    border: 1px solid var(--rim2);
    border-radius: 5px;
    padding: 2px 8px;
    color: #5D5A7E;
}

.mini-bars { display: flex; flex-direction: column; gap: 5px; }
.mbar-row  { display: flex; align-items: center; gap: 7px; }
.mbar-lbl  { font-size:.63rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; color:#3A3760; width:38px; flex-shrink:0; }
.mbar-track { flex:1; height:5px; background:#0A0A18; border-radius:999px; overflow:hidden; }
.mbar-fill  { height:100%; border-radius:999px; min-width:2px; }
.mbar-genre  { background: linear-gradient(90deg,#7B61FF,#A78BFA); }
.mbar-collab { background: linear-gradient(90deg,#0EA5E9,#38BDF8); }
.mbar-mood   { background: linear-gradient(90deg,#FFD166,#FBBF24); }
.mbar-num { font-size:.68rem; font-weight:600; color:#3A3760; width:28px; text-align:right; flex-shrink:0; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb { background: #2A2750; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--neon); }
"""

# ── THEME DESIGN ──────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#F5F3FF", c100="#EDE9FE", c200="#DDD6FE", c300="#C4B5FD",
        c400="#A78BFA", c500="#8B5CF6", c600="#7C3AED", c700="#6D28D9",
        c800="#5B21B6", c900="#4C1D95", c950="#2E1065",
    ),
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
)

# ── GRADIO BLOCKS ─────────────────────────────────────────────────────────────

with gr.Blocks(title="CineMatch – AI Movie Recommender") as demo:
    # Hero header
    gr.HTML("""
    <div class="hero">
        <div class="hero-grid"></div>
        <div class="hero-eyebrow"><div class="pulse"></div>Live · All models loaded</div>
        <h1>CineMatch</h1>
        <p class="hero-sub">
            Hybrid AI recommendation engine — genre intelligence,
            collaborative filtering &amp; real-time emotion detection.
        </p>
        <div class="tech-pills">
            <span class="tpill">🧮 TF-IDF Genre Vectors</span>
            <span class="tpill">🔢 SVD · 50 Latent Factors</span>
            <span class="tpill">🧠 DistilRoBERTa Emotion</span>
            <span class="tpill">🎬 MovieLens 32M</span>
        </div>
    </div>
    """)

    # Two-column body layout
    with gr.Row(elem_classes="body-wrap"):
        # Left — inputs
        with gr.Column(scale=0, min_width=360, elem_classes="left-panel"):
            gr.HTML('<div class="sect-label">Your Preferences</div>')

            movie_input = gr.Textbox(
                label="Favourite Movie",
                placeholder="e.g. The Dark Knight, Inception, Toy Story…",
                lines=1, max_lines=1,
            )
            mood_input = gr.Textbox(
                label="How Are You Feeling Right Now?",
                placeholder="e.g. I'm feeling excited and pumped up!",
                lines=2, max_lines=3,
            )
            submit_btn = gr.Button("✦  Get My Recommendations", variant="primary", size="lg")

            gr.HTML("""
            <div class="weight-card">
                <div style="font-size:.65rem;font-weight:700;letter-spacing:.12em;
                            text-transform:uppercase;color:#4A476A;margin-bottom:10px;">
                    Scoring Weights
                </div>
                <div class="wrow">
                    <span class="wrow-label"><i class="ld ld-genre"></i>Genre Similarity</span>
                    <span class="wrow-val" style="color:#A78BFA">40%</span>
                </div>
                <div class="wrow">
                    <span class="wrow-label"><i class="ld ld-collab"></i>Collaborative (SVD)</span>
                    <span class="wrow-val" style="color:#38BDF8">40%</span>
                </div>
                <div class="wrow">
                    <span class="wrow-label"><i class="ld ld-mood"></i>Mood Boost</span>
                    <span class="wrow-val" style="color:#FBBF24">20%</span>
                </div>
            </div>
            """)

            gr.HTML('<div class="sect-label" style="margin-top:4px">Quick Examples</div>')
            gr.Examples(
                examples=[
                    ["The Dark Knight", "I'm feeling thrilled and pumped!"],
                    ["Toy Story", "I'm happy and cheerful today 😊"],
                    ["Inception", "I'm feeling curious and thoughtful"],
                    ["The Notebook", "I'm a bit sad and nostalgic tonight"],
                    ["Interstellar", "I feel amazed and overwhelmed"],
                    ["The Avengers", "I'm angry, give me action!"],
                ],
                inputs=[movie_input, mood_input],
                label="",
            )

        # Right — output
        with gr.Column(scale=1, elem_classes="right-panel"):
            gr.HTML('<div class="sect-label">Your Recommendations</div>')
            output_html = gr.HTML(value=_idle())

    # Events Bindings
    submit_btn.click(
        fn=get_recommendations,
        inputs=[movie_input, mood_input],
        outputs=output_html,
        show_progress="full",
    )
    movie_input.submit(
        fn=get_recommendations,
        inputs=[movie_input, mood_input],
        outputs=output_html,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7865,
        css=CSS,
        theme=THEME,
    )