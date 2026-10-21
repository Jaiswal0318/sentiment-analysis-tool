
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# -------------------- Configuration --------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"

st.set_page_config(
    page_title="SentimentAI | Text Analyzer",
    page_icon="💬",
    layout="wide",
)

# -------------------- Load Trained Model --------------------
@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

# -------------------- Styling --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #101426, #171b32);
        color: #f3f4f6;
    }
    h1, h2, h3 {
        color: #f3f4f6;
    }
    div.stButton > button {
        background: #7c5cff;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background: #6241e8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Dashboard --------------------
st.title("💬 SentimentAI")
st.caption(
    "Understand the emotion behind your text with machine learning."
)
st.divider()

if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. "
        "Please run `python train_model.py` first."
    )
    st.stop()

model = load_model()

left, right = st.columns([3, 2])

with left:
    st.subheader("Analyze your text")
    user_text = st.text_area(
        "Enter a movie review, product review, or comment",
        placeholder=(
            "Example: The movie was fantastic, "
            "with an excellent story and brilliant acting!"
        ),
        height=180,
    )
    analyze = st.button(
        "Analyze Sentiment",
        use_container_width=True,
    )

with right:
    st.subheader("How it works")
    st.markdown(
        """
        - **NLP:** Processes the input text.
        - **TF-IDF:** Converts words into numerical features.
        - **Logistic Regression:** Predicts positive or negative sentiment.
        - **Probability chart:** Displays the model's class probabilities.
        """
    )

    st.info(
        "This version uses a trained movie-review dataset "
        "with two sentiment classes: positive and negative."
    )

# -------------------- Prediction Results --------------------
if analyze:
    if not user_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        prediction = model.predict([user_text])[0]
        probabilities = model.predict_proba([user_text])[0]
        classes = model.named_steps["classifier"].classes_

        confidence = max(probabilities) * 100

        st.divider()
        st.subheader("Analysis Results")

        col1, col2 = st.columns(2)
        col1.metric(
            "Predicted Sentiment",
            prediction.capitalize(),
        )
        col2.metric(
            "Model Confidence",
            f"{confidence:.1f}%",
        )

        if prediction == "pos":
            st.success("The model detected a positive sentiment.")
        else:
            st.error("The model detected a negative sentiment.")

        results = pd.DataFrame(
            {
                "Sentiment": [
                    label.capitalize() for label in classes
                ],
                "Probability (%)": probabilities * 100,
            }
        ).sort_values("Probability (%)", ascending=False)

        st.markdown("#### Sentiment probability distribution")
        st.bar_chart(
            results.set_index("Sentiment"),
            y="Probability (%)",
        )

        st.caption(
            "Evaluation accuracy: 84.50% on a held-out set of "
            "400 movie reviews. Individual predictions may be wrong. "
            "This model was trained on movie reviews and may be less "
            "reliable on product reviews, short text, or other domains."
        )

st.divider()
st.caption(
    "SentimentAI | Python • NLP • Scikit-learn • Streamlit"
)
