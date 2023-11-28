import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word not in stop_words])

    return text

def calculate_word_frequency(text_data):
    words = word_tokenize(text_data)
    word_freq = Counter(words)
    return word_freq

def display_top_words(word_freq, top_n):
    top_words = word_freq.most_common(top_n)
    print(f"\nTop {top_n} words and their frequencies:")
    for word, freq in top_words:
        print(f"{word}: {freq}")

def plot_bar_graph(word_freq, top_n):
    top_words = dict(word_freq.most_common(top_n))
    plt.figure(figsize=(10, 6))
    plt.bar(top_words.keys(), top_words.values(), color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example dataset
    data = {'feedback': ["Great product! Loved it.", "Not satisfied with the service.", "The app is fantastic."]}
    df = pd.DataFrame(data)

    # Preprocess text data
    df['processed_feedback'] = df['feedback'].apply(preprocess_text)

    # Concatenate all preprocessed texts
    all_text = ' '.join(df['processed_feedback'])

    # Calculate word frequency
    word_frequency = calculate_word_frequency(all_text)

    # Get user input for top N words
    top_n = int(input("Enter the value of N for top words: "))

    # Display top N words and frequencies
    display_top_words(word_frequency, top_n)

    # Plot bar graph
    plot_bar_graph(word_frequency, top_n)
