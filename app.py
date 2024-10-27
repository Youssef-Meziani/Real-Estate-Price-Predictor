from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

import pickle

app = Flask(__name__)

usa_housing_data = pd.read_csv('USA_Housing.csv')
X = usa_housing_data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                      'Avg. Area Number of Bedrooms', 'Area Population']]
Y = usa_housing_data['Price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

model = LinearRegression()
model.fit(X_train, Y_train)


def predict_price(Income, House_Age, Number_of_Rooms, Number_of_Bedrooms, Population):
    data = {'Avg. Area Income': [Income],
            'Avg. Area House Age': [House_Age],
            'Avg. Area Number of Rooms': [Number_of_Rooms],
            'Avg. Area Number of Bedrooms': [Number_of_Bedrooms],
            'Area Population': [Population]}

    return model.predict(pd.DataFrame(data))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    Income = float(request.form['income'])
    House_Age = float(request.form['age'])
    Number_of_Rooms = float(request.form['rooms'])
    Number_of_Bedrooms = float(request.form['bedrooms'])
    Population = float(request.form['population'])

    prediction = predict_price(Income, House_Age, Number_of_Rooms, Number_of_Bedrooms, Population)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Estimated Price: {output}$')


if __name__ == "__main__":
    app.run(debug=True)
