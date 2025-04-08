import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import normalize
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

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

# Load models at startup for faster inference
tfidf_vectorizer, svd, knn_model, recipes_df = load_models()

def get_recipe_recommendations(ingredients_list, top_n=5):
    # Check if any of the models or data failed to load
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
    columns_to_return = [
        'TranslatedRecipeName', 'TranslatedIngredients', 'TotalTimeInMins', 
        'TranslatedInstructions', 'Cuisine', 'URL', 'image-url'
    ]
    
    # Only include columns that exist in the dataframe
    valid_columns = [col for col in columns_to_return if col in recipes_df.columns]
    
    recommended_recipes = recipes_df.iloc[original_indices][valid_columns].to_dict(orient='records')
    
    return recommended_recipes

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "active",
        "message": "Recipe Recommendation API is running. Use POST /recommend endpoint with ingredients."
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        
        if not data or 'ingredients' not in data:
            return jsonify({"error": "Please provide ingredients in the request body"}), 400
        
        ingredients = data['ingredients']
        if not isinstance(ingredients, list):
            return jsonify({"error": "Ingredients must be provided as a list"}), 400
        
        top_n = data.get('top_n', 3)  # Default to 3 recommendations
        
        recommendations = get_recipe_recommendations(ingredients, top_n=top_n)
        
        if recommendations is None:
            return jsonify({"error": "Failed to get recommendations"}), 500
        
        return jsonify({
            "ingredients": ingredients,
            "recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)