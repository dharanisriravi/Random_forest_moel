from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and columns
model = pickle.load(open('job_model.pkl', 'rb'))
columns = pickle.load(open('product_mapping.pkl', 'rb'))

# Load product info
products = pd.read_csv('products.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    age = int(request.form['age'])
    gender = request.form['gender']
    job = request.form['job']
    income = int(request.form['income'])
    marital_status = request.form['marital_status']

    user_input = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'job': job,
        'income': income,
        'marital_status': marital_status
    }])

    user_encoded = pd.get_dummies(user_input)
    user_encoded = user_encoded.reindex(columns=columns, fill_value=0)

    prediction = model.predict(user_encoded)[0]
    product = products[products['product_id'] == prediction].iloc[0]

    return render_template('result.html', name=product['product_name'], category=product['category'])

if __name__ == '__main__':
    app.run(debug=True)
