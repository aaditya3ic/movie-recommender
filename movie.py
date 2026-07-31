# ==============================================================================
# PROJECT 4: MOVIE RECOMMENDATION SYSTEM (Content-Based Filtering)
# ==============================================================================

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------------------
# STEP 1: Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Movie Recommender AI", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")
st.write("A Content-Based AI that suggests movies purely by comparing their text descriptions.")

# ------------------------------------------------------------------------------
# STEP 2 & 3: Create Dataset and Calculate Similarity (Cached for Performance)
# ------------------------------------------------------------------------------
@st.cache_data
def load_data_and_model():
    # Step 4.1: Create the Dataset
    movies_df = pd.DataFrame({
        "title": [
            "Interstellar", "Inception", "The Martian", "Arrival",
            "The Matrix", "Avatar", "Titanic", "The Notebook",
            "Avengers: Endgame", "Iron Man", "Jurassic Park", "The Dark Knight"
        ],
        "description": [
            "space science fiction astronauts future adventure",
            "science fiction dreams technology thriller mind bending",
            "space science fiction astronaut survival mars adventure",
            "science fiction aliens language space mystery",
            "science fiction technology artificial intelligence action",
            "science fiction space aliens adventure fantasy",
            "romance drama ship ocean historical tragedy",
            "romance relationship love drama emotional",
            "superhero action marvel time travel adventure",
            "superhero action technology marvel engineering",
            "dinosaurs science adventure action island",
            "superhero action crime batman thriller"
        ]
    })

    # Step 4.2: Convert Descriptions into Numbers (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words="english")
    movie_matrix = vectorizer.fit_transform(movies_df["description"])

    # Step 4.3: Calculate Cosine Similarity
    sim_matrix = cosine_similarity(movie_matrix)
    
    return movies_df, sim_matrix

movies, similarity_matrix = load_data_and_model()

# ------------------------------------------------------------------------------
# STEP 4: Build the Recommendation Logic
# ------------------------------------------------------------------------------
def recommend_movies(movie_title, number_of_recommendations=5):
    if movie_title not in movies["title"].values:
        return None

    # Find the row index of the selected movie
    movie_index = movies.index[movies["title"] == movie_title][0]

    # Get similarity scores for this movie compared to all others
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    # Sort from highest similarity to lowest
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Remove the movie's comparison with itself (which is always 1.0)
    similarity_scores = [item for item in similarity_scores if item[0] != movie_index]

    # Build the final list of recommendations
    recommendations = []
    for index, score in similarity_scores[:number_of_recommendations]:
        recommendations.append({
            "Movie Title": movies.iloc[index]["title"],
            "Match Score": f"{round(score * 100, 1)}%"
        })

    return pd.DataFrame(recommendations)

# ------------------------------------------------------------------------------
# STEP 5: Interactive UI for the Presentation
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ System Internal Stats")
st.sidebar.write(f"**Total Movies:** {len(movies)}")
st.sidebar.write(f"**Matrix Shape:** {similarity_matrix.shape}")
st.sidebar.write("---")
st.sidebar.subheader("Raw Dataset")
st.sidebar.dataframe(movies, hide_index=True)

st.subheader("1. Select a Movie You Like")
selected_movie = st.selectbox("Choose a film from the database:", movies["title"].values)

st.subheader("2. AI Recommendations")
if st.button("🔍 Find Similar Movies", type="primary"):
    with st.spinner("Calculating vector similarities..."):
        results_df = recommend_movies(selected_movie, number_of_recommendations=3)
        
        if results_df is not None:
            st.success(f"Because you liked **{selected_movie}**, we recommend:")
            st.table(results_df)
        else:
            st.error("Movie not found in database.")

# ------------------------------------------------------------------------------
# STEP 6: Educational Resources & Course Wrap-up
# ------------------------------------------------------------------------------
st.markdown("---")
st.header("📚 Presentation Talking Points & Course Wrap-Up")

with st.expander("✅ Prerequisites: How This Works"):
    st.write("""
    - **TF-IDF:** The same numeric text-conversion technique from Project 1, but now used to measure *similarity* between items instead of classifying them into categories.
    - **Cosine Similarity:** A way of measuring how similar two vectors (lists of numbers) are, based on the *angle* between them rather than their absolute size. A score of `1` means the two vectors point in exactly the same direction (maximum similarity); a score of `0` means they are unrelated.
    - **Content-based vs. Collaborative filtering:** Content-based systems compare item descriptions (what we built today); collaborative systems compare patterns across many users' behavior ("people who liked what you liked also liked...").
    """)

with st.expander("🤔 What Did the AI Actually Do?"):
    st.write("""
    We did not hard-code: *"If the user likes Interstellar, recommend The Martian."*
    
    Instead, the system compared numerical representations of movie descriptions and found the closest matches mathematically. Real recommendation platforms (like Netflix) typically combine several additional signals: user behavior, watch history, ratings, similar users, content, context, and deep learning.
    """)

with st.expander("💼 Interview Corner"):
    st.markdown("""
    1. What's the difference between content-based and collaborative filtering?
    2. Why is cosine similarity preferred over Euclidean distance for text vectors?
    3. What is the "cold start problem," and which approach handles it better?
    4. How would you evaluate a recommender system's quality (offline vs. online metrics)?
    5. What's one weakness of a purely content-based system? *(Hint: think about the filter bubble effect and a lack of serendipity.)*
    """)

with st.expander("🔥 Final Challenge — Compare the Four AI Projects"):
    st.markdown("""
    | Application | Input | Technique | Output |
    |---|---|---|---|
    | **Sentiment Analysis** | Text | NLP + ML | Sentiment |
    | **Image Recognition** | Image | Deep Learning | Object Class |
    | **Face Detection** | Image | Computer Vision | Face Location |
    | **Recommendation** | Movie Metadata | Similarity | Recommendations |
    
    **Challenge Questions:**
    1. What type of data does each project use?
    2. Which project is an example of supervised learning?
    3. Which project uses similarity instead of classification?
    4. Which project uses a pretrained deep learning model?
    5. If you had to add one more real-world AI project to this list, what would it be, and why?
    """)

with st.expander("🚀 Real AI Products & 🧠 Final Takeaway"):
    st.markdown("""
    **How These Ideas Appear in Real AI Products:**
    - **AI Assistants:** NLP + Deep Learning + Multimodal Models
    - **Google Photos:** Computer Vision + Image Understanding
    - **Netflix / Spotify:** Recommendation Models
    - **Autonomous Vehicles:** Computer Vision + Deep Learning + Sensor Fusion
    
    *Real AI products are usually systems made up of multiple models, data pipelines, software components, and business logic working together — not a single algorithm.*
    
    **🧠 Final Takeaway:**
    The common pattern underneath all four apps built today is:
    **Real-World Data → Numerical Representation → Algorithm/Model → Prediction, Detection, or Recommendation.**
    That is the foundation of practical AI development with Python.
    """)

with st.expander("📌 Optional Next Projects"):
    st.markdown("""
    1. Spam Email Classifier
    2. Fraud Detection
    3. Speech-to-Text AI
    4. AI Image Caption Generator
    5. Object Detection with YOLO
    6. Handwritten Digit Recognition
    7. Chatbot with Intent Classification
    8. Customer Churn Prediction
    9. AI-Powered Resume Screening
    10. Generative AI with Large Language Models
    
    **Suggested progression:** Classical ML → NLP → Computer Vision → Deep Learning → Generative AI → LLMs and Multimodal AI
    """)