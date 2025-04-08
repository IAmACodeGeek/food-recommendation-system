# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string
import time
import pickle
import os

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

print("Starting Recipe Recommendation System...")

# Load Dataset from local directory
file_path = './Cleaned_Indian_Food_Dataset.csv'  # Change this to your specific file name in current directory
print(f"Loading dataset from {file_path}...")

try:
    recipes_df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {recipes_df.shape[0]} rows and {recipes_df.shape[1]} columns.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the file path.")
    exit(1)

# Data Preprocessing
print("\nPerforming data preprocessing...")

# Check for missing values
print("\nMissing Values:")
print(recipes_df.isnull().sum())

# Remove NaN values
print(f"Number of rows before dropping NaN values: {len(recipes_df)}")
recipes_df.dropna(inplace=True)
print(f"Number of rows after dropping NaN values: {len(recipes_df)}")

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Join tokens back into a string
    return ' '.join(tokens)

# Apply text preprocessing to relevant columns
print("Applying text preprocessing...")
recipes_df['name_processed'] = recipes_df['TranslatedRecipeName'].apply(preprocess_text)
recipes_df['ingredients_processed'] = recipes_df['TranslatedIngredients'].apply(preprocess_text)
recipes_df['instructions_processed'] = recipes_df['TranslatedInstructions'].apply(preprocess_text)
recipes_df['cuisine_processed'] = recipes_df['Cuisine'].apply(preprocess_text)

# Process the cleaned ingredients column
if 'Cleaned-Ingredients' in recipes_df.columns:
    recipes_df['cleaned_ingredients_processed'] = recipes_df['Cleaned-Ingredients'].apply(preprocess_text)
else:
    print("Warning: 'Cleaned-Ingredients' column not found. Using 'TranslatedIngredients' instead.")
    recipes_df['cleaned_ingredients_processed'] = recipes_df['ingredients_processed']

# Create a combined text field for searching
recipes_df['combined_text'] = (
    recipes_df['name_processed'] + ' ' +
    recipes_df['cuisine_processed'] + ' ' +
    recipes_df['ingredients_processed'] + ' ' +
    recipes_df['instructions_processed'] + ' ' +
    recipes_df['cleaned_ingredients_processed']
)

print("Text preprocessing completed.")

# Create testing set (20% of data) but keep the full dataset for training
print("\nCreating training and testing sets...")
X = recipes_df['combined_text']
y = recipes_df['TranslatedRecipeName']  # Use the original recipe name as the target

# Create a test set (20% of the data)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use the FULL dataset for training (not just the remaining 80%)
X_train = X
y_train = y

print(f"Training set size: {len(X_train)} (full dataset)")
print(f"Testing set size: {len(X_test)} (20% of dataset)")

# Apply TF-IDF vectorization on the full dataset
print("\nApplying TF-IDF vectorization to the full dataset...")
start_time = time.time()
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_full_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"TF-IDF vectorization completed in {time.time() - start_time:.2f} seconds")
print(f"TF-IDF matrix shape: {X_full_tfidf.shape}")

# Apply dimensionality reduction with SVD
print("\nApplying SVD dimensionality reduction...")
start_time = time.time()
n_components = 300  # Number of components to keep
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_full_svd = svd.fit_transform(X_full_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

# Normalize the vectors
X_full_svd_norm = normalize(X_full_svd)
X_test_svd_norm = normalize(X_test_svd)

print(f"SVD reduction completed in {time.time() - start_time:.2f} seconds")
print(f"Reduced dimensions: {X_full_svd.shape}")
explained_variance = svd.explained_variance_ratio_.sum()
print(f"Explained variance ratio: {explained_variance:.4f}")

# Train KNN model on the full dataset
print("\nTraining KNN model on full dataset...")
start_time = time.time()
knn_model = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine')
knn_model.fit(X_full_svd_norm)
print(f"KNN model training completed in {time.time() - start_time:.2f} seconds")

# Function to search for recipes using the combined approach
def search_recipes(query, top_n=3):
    # Preprocess the query
    processed_query = preprocess_text(query)
    
    # Transform the query using the same pipeline (TF-IDF -> SVD -> Normalize)
    query_tfidf = tfidf_vectorizer.transform([processed_query])
    query_svd = svd.transform(query_tfidf)
    query_svd_norm = normalize(query_svd)
    
    # Find the nearest neighbors
    distances, indices = knn_model.kneighbors(query_svd_norm, n_neighbors=top_n)
    
    # Convert indices to original dataframe indices
    original_indices = indices.flatten()
    
    # Return the top N recipes
    return recipes_df.iloc[original_indices][['TranslatedRecipeName', 'TranslatedIngredients', 'TotalTimeInMins', 'TranslatedInstructions', 'Cuisine']]

# Evaluation function
def evaluate_model(X_test, y_test, top_n=3, num_samples=None):
    print(f"\nEvaluating model with top-{top_n} predictions...")
    
    if num_samples:
        # Select a subset of test samples
        test_indices = X_test.index[:num_samples]
    else:
        test_indices = X_test.index
    
    total = len(test_indices)
    correct = 0
    
    start_time = time.time()
    
    for i, test_idx in enumerate(test_indices):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{total} test samples...")
        
        # Get the true recipe name and ingredients
        true_name = recipes_df.loc[test_idx, 'TranslatedRecipeName']
        test_ingredients = recipes_df.loc[test_idx, 'TranslatedIngredients']
        
        # Get predictions using ingredients as query
        results = search_recipes(test_ingredients, top_n=top_n)
        
        # Check if the true recipe name is among the predictions
        if true_name in results['TranslatedRecipeName'].values:
            correct += 1
    
    accuracy = correct / total
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    print(f"Accuracy (top-{top_n}): {accuracy:.4f} ({correct}/{total})")
    
    return accuracy

# Test the model with a sample query
sample_query = "karela (bitter gourd pavakkai), red chilli powder, gram flour (besan)"
print(f"\nSample query: '{sample_query}'")
print("Top results:")
result = search_recipes(sample_query, top_n=3)

for index, row in result.iterrows():
    print('=' * 100)
    print(f"Recipe: {row['TranslatedRecipeName']}")
    print(f"Cuisine: {row['Cuisine']}")
    print(f"Ingredients: {row['TranslatedIngredients']}")
    print(f"Cooking time: {row['TotalTimeInMins']} minutes")
    print(f"Instructions: {row['TranslatedInstructions'][:200]}...")
print('=' * 100)

# Evaluate the model with top-3 predictions
# Using a smaller subset for quick validation
num_eval_samples = len(X_test)  # Increase this for more thorough evaluation
accuracy = evaluate_model(X_test, y_test, top_n=1, num_samples=num_eval_samples)

# Save the model
print("\nSaving models...")
os.makedirs('models', exist_ok=True)
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('models/svd_reducer.pkl', 'wb') as f:
    pickle.dump(svd, f)
with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

print("\nModel training and evaluation completed!")