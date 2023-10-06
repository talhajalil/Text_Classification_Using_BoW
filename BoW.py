from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier

# Define a function to preprocess text and remove stopwords
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

training_data = {
    "Politics": [
        "The government announced new policies for economic reforms.",
        "The opposition parties criticize the budget proposal.",
        "Political leaders are attending an international summit.",
        "A new trade agreement has been reached between two countries.",
        "Election season is approaching, and candidates are campaigning.",
        "The parliament passed a new law on healthcare reform.",
        "Political analysts are predicting the outcome of the upcoming elections.",
        "The president's speech addressed key economic challenges.",
        "A major political rally will be held in the city center.",
        "Foreign policy issues were discussed during the diplomatic meeting."
    ],
    "Sports": [
        "The home team won the championship after a thrilling match.",
        "The star player has been traded to a rival team.",
        "A new sports stadium is being built in the city.",
        "The coach announced the team's new training schedule.",
        "The sports event attracted a large audience from all over the world.",
        "The Olympic Games brought together athletes from different countries.",
        "A world record was set in the 100m sprint at the track and field event.",
        "The baseball team won the league championship.",
        "A famous tennis player announced their retirement from professional sports.",
        "The soccer team is preparing for an important international tournament."
    ],
    "Technology": [
        "A new smartphone with advanced features has been launched.",
        "Researchers have made a breakthrough in artificial intelligence.",
        "The technology company announced record profits this quarter.",
        "A software update is available for your computer.",
        "A tech startup received funding for its innovative project.",
        "The latest tech gadgets were showcased at the consumer electronics expo.",
        "Scientists developed a new renewable energy technology.",
        "A cybersecurity firm detected a major data breach.",
        "Tech enthusiasts are eagerly awaiting the release of the new gaming console.",
        "Advancements in quantum computing are shaping the future of technology."
    ]
}


# Test Data with class information
test_data = {
    "Test Article 1 - Technology": "A new study reveals the impact of technology on modern society.",
    "Test Article 2 - Politics": "The political leaders are discussing a new trade deal.",
    "Test Article 3 - Sports": "The sports event attracted a record number of spectators."
}

# Create a vocabulary from training data
vocabulary = set()
for category, articles in training_data.items():
    for article in articles:
        preprocessed_text = preprocess_text(article)
        words = preprocessed_text.split()
        vocabulary.update(words)

# Create a mapping from words to indices
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Generate BoW vectors for training data
X_train = []
y_train = []
for category, articles in training_data.items():
    for article in articles:
        preprocessed_text = preprocess_text(article)
        word_counts = Counter(preprocessed_text.split())
        bow_vector = [word_counts[word] for word in vocabulary]
        X_train.append(bow_vector)
        y_train.append(category)

# Train a K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# Generate BoW vectors for test data
X_test = []
true_labels = []  # Store true labels for accuracy calculation

for article_title, article_text in test_data.items():
    preprocessed_text = preprocess_text(article_text)
    word_counts = Counter(preprocessed_text.split())
    bow_vector = [word_counts[word] for word in vocabulary]
    X_test.append(bow_vector)

for article_title in test_data.keys():
    if "Technology" in article_title:
        true_labels.append("Technology")
    elif "Politics" in article_title:
        true_labels.append("Politics")
    elif "Sports" in article_title:
        true_labels.append("Sports")

# Classify the test articles
predicted_labels = knn_classifier.predict(X_test)

# Calculate accuracy
correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
accuracy = correct_predictions / len(true_labels) * 100

# Print the results
for article_title, predicted_category in zip(test_data.keys(), predicted_labels):
    print(f"{article_title}: Predicted Category: {predicted_category}")

# Print accuracy
print(f"Accuracy: {accuracy:.2f}%")
