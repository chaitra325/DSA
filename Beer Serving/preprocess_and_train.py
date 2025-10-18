import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import pickle
import os

# 1. Data Preprocessing
df = pd.read_csv("beer_servings.csv")
df = df.drop_duplicates()
df.replace(['unknown', 'Unknown', 'UNK', 'N/A', 'na', '-', ''], np.nan, inplace=True)
for col in ['beer_servings', 'spirit_servings', 'wine_servings', 'total_litres_of_pure_alcohol']:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df[
    (df['beer_servings'] >= 0) &
    (df['spirit_servings'] >= 0) &
    (df['wine_servings'] >= 0) &
    (df['total_litres_of_pure_alcohol'] >= 0) &
    (df['total_litres_of_pure_alcohol'] <= 20)
]
for col in ['beer_servings', 'spirit_servings', 'wine_servings', 'total_litres_of_pure_alcohol']:
    df[col].fillna(df[col].mean(), inplace=True)
for col in ['country', 'continent']:
    df[col].fillna(df[col].mode()[0], inplace=True)
le_country = LabelEncoder()
le_continent = LabelEncoder()
df['country_encoded'] = le_country.fit_transform(df['country'])
df['continent_encoded'] = le_continent.fit_transform(df['continent'])
df.to_csv("beer_servings_cleaned.csv", index=False)

# 2. EDA Infographics
os.makedirs("static", exist_ok=True)
plt.figure(figsize=(7, 4))
sns.histplot(df['beer_servings'])
plt.title("Beer Servings Distribution")
plt.savefig("static/infographic.png")
plt.close()
sns.countplot(x='continent', data=df)
plt.title("Continent Distribution")
plt.savefig("static/continent_bar.png")
plt.close()

# 3. Model Training
X = df[['beer_servings', 'spirit_servings', 'wine_servings', 'country_encoded', 'continent_encoded']]
y = df['total_litres_of_pure_alcohol']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_r2 = r2_score(y_test, lr.predict(X_test))

rf = RandomForestRegressor(random_state=42)
param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7]}
gs = GridSearchCV(rf, param_grid, scoring='r2', cv=3)
gs.fit(X_train, y_train)
rf_r2 = r2_score(y_test, gs.predict(X_test))
print(f"Linear Regression R2: {lr_r2:.3f}")
print(f"Random Forest R2: {rf_r2:.3f}")
print("Best RF Params:", gs.best_params_)

if rf_r2 > lr_r2:
    pickle.dump(gs.best_estimator_, open('model.pkl', 'wb'))
    print("Random Forest selected and saved.")
else:
    pickle.dump(lr, open('model.pkl', 'wb'))
    print("Linear Regression selected and saved.")
pickle.dump(le_country, open('le_country.pkl', 'wb'))
pickle.dump(le_continent, open('le_continent.pkl', 'wb'))
print("Preprocessing, EDA, training & pickling done.")