import json
import numpy as np
import random
import nltk
import tensorflow_hub as hub 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Universal Sentence Encoder (USE)
print("Loading Universal Sentence Encoder...")
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load dataset from JSON file
with open("ok11.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Data augmentation functions
def synonym_replacement(sentence, n=2):
    words = sentence.split()
    if not words:  # Check if the list is empty
        return sentence  # Return the original sentence if no words are left
    new_words = words.copy()
    for _ in range(n):
        word = random.choice(words)
        synonyms = []
        for syn in wordnet.synsets(word):  # Get all synsets for the word
            for lemma in syn.lemmas():    # Get all lemmas for each synset
                synonyms.append(lemma.name())
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if w == word else w for w in new_words]
    return ' '.join(new_words)

def random_insertion(sentence, n=2):
    words = sentence.split()
    if not words:  # Check if the list is empty
        return sentence  # Return the original sentence if no words are left
    for _ in range(n):
        word = random.choice(words)
        synonyms = []
        for syn in wordnet.synsets(word):  # Get all synsets for the word
            for lemma in syn.lemmas():    # Get all lemmas for each synset
                synonyms.append(lemma.name())
        if synonyms:
            synonym = random.choice(synonyms)
            words.insert(random.randint(0, len(words)), synonym)
    return ' '.join(words)

# Prepare training data with augmentation
all_patterns = []
all_tags = []
labels = []
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Preprocess the original pattern
        words = word_tokenize(pattern.lower())
        filtered_words = [lemmatizer.lemmatize(w) for w in words if w.isalnum() and w not in stop_words]
        original_sentence = " ".join(filtered_words)
        if original_sentence:  # Only add non-empty sentences
            all_patterns.append(original_sentence)
            labels.append(intent['tag'])
            
            # Generate augmented sentences
            augmented_sentence_1 = synonym_replacement(original_sentence)  # Synonym replacement
            augmented_sentence_2 = random_insertion(original_sentence)     # Random insertion
            
            # Add augmented sentences to the dataset
            if augmented_sentence_1:  # Only add non-empty sentences
                all_patterns.append(augmented_sentence_1)
                labels.append(intent['tag'])
            if augmented_sentence_2:  # Only add non-empty sentences
                all_patterns.append(augmented_sentence_2)
                labels.append(intent['tag'])
        
    if intent['tag'] not in all_tags:
        all_tags.append(intent['tag'])

# Convert text to numerical features using USE
X = use_model(all_patterns).numpy()
y = np.array([all_tags.index(tag) for tag in labels])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build neural network model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    # Dense(32, activation='relu'),  # Added a smaller layer
    # Dropout(0.2),
    Dense(len(all_tags), activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=80, batch_size=32, verbose=1)
model.save("chatbot_model.keras")
print("Model saved successfully!")
# Evaluate the model
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=all_tags, yticklabels=all_tags)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Chatbot function
def chatbot_response(text):
    # Preprocess input
    words = word_tokenize(text.lower())
    processed_text = " ".join([lemmatizer.lemmatize(w) for w in words if w.isalnum() and w not in stop_words])
    
    # Convert input to USE embedding
    X_input = use_model([processed_text]).numpy()
    
    # Predict intent
    prediction = model.predict(X_input)
    tag_index = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Confidence threshold
    if confidence < 0.7:  
        return "I'm not sure I understand. Can you please rephrase?"
    
    # Get response
    tag = all_tags[tag_index]
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand that."

# Test the chatbot
print("Chatbot: Hi! I'm your car assistant. How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)


