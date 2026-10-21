
import nltk
import pickle
from pathlib import Path

from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"

# Download the movie reviews dataset if needed
nltk.download("movie_reviews")

# Load labeled reviews
documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

texts = [text for text, label in documents]
labels = [label for text, label in documents]

print(f"Loaded {len(documents)} reviews.")
print(f"Class distribution: { {label: labels.count(label) for label in set(labels)} }")

# Separate training and test data
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# Build the machine-learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# Train only on the training split
print("\nTraining the model...")
model.fit(X_train, y_train)

# Evaluate on unseen test reviews
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n--- Model Evaluation ---")
print(f"Test accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(classification_report(y_test, predictions))

# Save the trained model
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with MODEL_PATH.open("wb") as file:
    pickle.dump(model, file)

print(f"\nModel saved to: {MODEL_PATH}")
