import pandas as pd
import numpy as np
import pickle
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import normalize

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

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

def load_models():
    print("Loading models...")
    
    try:
        # Load TF-IDF vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        # Load SVD and KNN models
        with open('models/svd_reducer.pkl', 'rb') as f:
            svd = pickle.load(f)
        with open('models/knn_model.pkl', 'rb') as f:
            knn_model = pickle.load(f)
        print("Models loaded successfully")
        
        # Load the recipes dataset
        recipes_df = pd.read_csv('./Cleaned_Indian_Food_Dataset.csv')
        print(f"Dataset loaded with {recipes_df.shape[0]} recipes")
        
        return tfidf_vectorizer, svd, knn_model, recipes_df
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None, None, None

def get_recipe_recommendations(ingredients_list, top_n=5):
    # Load models
    tfidf_vectorizer, svd, knn_model, recipes_df = load_models()
    print(recipes_df.columns)
    # Check if any of the models or data failed to load (using separate checks)
    if tfidf_vectorizer is None or svd is None or knn_model is None or recipes_df is None:
        print("Error: Failed to load necessary models or data")
        return None
    
    # Convert ingredients list to a string
    ingredients_string = ", ".join(ingredients_list)
    print(f"Searching for recipes with: {ingredients_string}")
    
    # Preprocess the query
    processed_query = preprocess_text(ingredients_string)
    
    # Transform the query using the same pipeline (TF-IDF -> SVD -> Normalize)
    query_tfidf = tfidf_vectorizer.transform([processed_query])
    query_svd = svd.transform(query_tfidf)
    query_svd_norm = normalize(query_svd)
    
    # Find the nearest neighbors
    distances, indices = knn_model.kneighbors(query_svd_norm, n_neighbors=top_n)
    original_indices = indices.flatten()
    
    # Return the top N recipes
    recommended_recipes = recipes_df.iloc[original_indices][
        ['TranslatedRecipeName', 'TranslatedIngredients', 'TotalTimeInMins', 
         'TranslatedInstructions', 'Cuisine', 'URL', 'image-url']
    ]
    
    return recommended_recipes

def main():
    print("Recipe Recommendation System")
    print("===========================")
    
    while True:
        # Get ingredient input from user
        print("\nEnter ingredients (comma-separated) or 'quit' to exit:")
        user_input = input("> ")
        
        if user_input.lower() == 'quit':
            break
        
        # Convert input to list of ingredients
        ingredients = [ing.strip() for ing in user_input.split(',')]
        
        # Get recommendations
        recommendations = get_recipe_recommendations(ingredients, top_n=3)
        
        if recommendations is not None:
            print("\nRecommended Recipes:")
            for index, row in recommendations.iterrows():
                print('=' * 80)
                print(f"Recipe: {row['TranslatedRecipeName']}")
                print(f"Cuisine: {row['Cuisine']}")
                print(f"URL: {row['URL']}")
                print(f"Ingredients: {row['TranslatedIngredients']}")
                print(f"Cooking time: {row['TotalTimeInMins']} minutes")
                print(f"Instructions: {row['TranslatedInstructions']}...")
            print('=' * 80)

if __name__ == "__main__":
    main()