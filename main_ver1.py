""" Program to recommend movies """

import re
from flask import Flask, render_template, request, flash
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
app.secret_key = 'secretkey'

# Load the ratings and movies datasets
ratings = pd.read_csv('Desktop/Dataset-splaSH/ratings.csv')
movies = pd.read_csv('Desktop/Dataset-splaSH/movies.csv')

# Create a user-item ratings matrix
ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Convert the ratings matrix to a sparse matrix for memory efficiency
ratings_sparse = sparse.csr_matrix(ratings_matrix)

# Calculate item-item cosine similarity
item_similarity = cosine_similarity(ratings_sparse.T)

# Create a DataFrame for item similarities
item_similarity_df = pd.DataFrame(item_similarity, index=ratings_matrix.columns, 
                                    columns=ratings_matrix.columns)

# Function to get movie recommendations for a given movie using weighted sum and regression
def get_movie_recommendations(movie_title, num_recommendations=5):
    """ Function to find recommended movies """

    # Get the similarity scores for the given movie
    movie_similarity = item_similarity_df[movie_title]

    # Sort the movies by similarity score in descending order
    recommended_movies = movie_similarity.sort_values(ascending=False)

    # Exclude the input movie itself
    recommended_movies = recommended_movies.drop(movie_title)

    # Get the top N recommended movies
    top_recommendations = recommended_movies.head(num_recommendations)

    # Retrieve movie titles based on movieId
    recommended_movie_titles = movies[movies['movieId'].isin(top_recommendations.index)]['title']

    return recommended_movie_titles, top_recommendations.index.tolist()

# Function to predict a user's rating for a movie using regression
def predict_user_rating(user_id, movie_id):
    """ Function to find new user rating """
    # Get the user's ratings
    user_ratings = ratings_matrix.loc[user_id]

    # Get similarity scores for the movie
    movie_similarity = item_similarity_df[movie_id]

    # Calculate a weighted sum of user ratings based on movie similarity
    weighted_sum = (user_ratings * movie_similarity).sum()

    # Perform regression to refine the prediction
    x = item_similarity_df[movie_id].values.reshape(-1, 1)
    y = user_ratings.values

    regression_model = LinearRegression()
    regression_model.fit(x, y)

    # Predict the user's rating using regression
    predicted_rating = regression_model.predict([[weighted_sum]])

    # Create a MinMaxScaler object with range 1 to 5
    #scaler = MinMaxScaler(feature_range=(1, 5))

    # Reshape the predicted rating for scaling
    #predicted_rating_reshaped = predicted_rating.reshape(-1, 1)

    # Normalize the predicted rating
    #normalized_rating = scaler.fit_transform(predicted_rating_reshaped)

    # Reshape it back to get a single value
    #normalized_rating = normalized_rating.reshape(-1)[0]

    # Ensure the normalized rating is within the range of 1 to 5
    # normalized_rating = min(max(normalized_rating, 1), 5)

    return predicted_rating[0]

def clean_text(text):
    """Cleans the input text by removing non-alphanumeric characters and converting to lowercase"""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    """ Driving Code """
    if request.method == 'POST':
        user_input = request.form.get('movie')
        user_input_clean = clean_text(user_input)

        try:
            idx = movies[movies['title_clean'].str.contains(user_input_clean)].iloc[0]["movieId"]
        except IndexError:
            flash('Nothing found.')
            return render_template('index.html')

        recommendations, recommended_movie_id = get_movie_recommendations(idx)

        # Calculate predicted ratings for the recommendations
        predicted_ratings = [predict_user_rating(1, movie_id) for movie_id in recommended_movie_id]

        # Retrieve movie titles based on movieId
        recommended_movie_titles = movies[movies['movieId'].isin(recommended_movie_id)]['title']

        # Combine movie titles, movie IDs, years, and predicted ratings
        movie_ratings = list(zip(recommendations, recommended_movie_titles, predicted_ratings))

        return render_template('index.html', movie_ratings=movie_ratings)
    return render_template('index.html')

if __name__ == '__main__':
    movies['title_clean'] = movies['title'].apply(clean_text)
    app.run(debug=True)
