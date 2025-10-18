from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
le_country = pickle.load(open('le_country.pkl', 'rb'))
le_continent = pickle.load(open('le_continent.pkl', 'rb'))
df = pd.read_csv("beer_servings_cleaned.csv")

countries = sorted(df['country'].unique())
continents = sorted(df['continent'].unique())

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', countries=countries, continents=continents)

@app.route('/predict', methods=['POST'])
def predict():
    beer = float(request.form['beer_servings'])
    spirit = float(request.form['spirit_servings'])
    wine = float(request.form['wine_servings'])
    country = request.form['country']
    continent = request.form['continent']

    country_encoded = le_country.transform([country])[0]
    continent_encoded = le_continent.transform([continent])[0]
    X_input = np.array([[beer, spirit, wine, country_encoded, continent_encoded]])
    pred = model.predict(X_input)[0]
    return render_template('result.html', prediction=round(pred,2),
                           beer=beer, spirit=spirit, wine=wine, country=country, continent=continent)

if __name__ == '__main__':
    app.run(debug=True)