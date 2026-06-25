from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = "secretkey"

# Load trained model
model = joblib.load("lung_cancer_stacking_model.pkl")

# ============================
# 1. Home Page
# ============================
@app.route('/')
def home():
    return render_template("home.html")

# ============================
# 2. Register / Login Pages
# (Temporary – no DB yet)
# ============================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        flash("Registration Successful (Temporary)!", "success")
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        flash("Login Successful (Temporary)!", "success")
        return redirect(url_for('predict'))
    return render_template("login.html")

# ============================
# 3. Prediction Page
# ============================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [
                int(request.form['gender']),
                int(request.form['age']),
                int(request.form['smoking']),
                int(request.form['yellow_f']),
                int(request.form['anxiety']),
                int(request.form['peer_pre']),
                int(request.form['chronic']),
                int(request.form['fatigue']),
                int(request.form['allergy']),
                int(request.form['wheezing']),
                int(request.form['alcohol']),
                int(request.form['coughing']),
                int(request.form['shortnes']),
                int(request.form['swallow']),
                int(request.form['chest_pai'])
            ]
            features = np.array(features).reshape(1, -1)

            prediction = model.predict(features)[0]
            result = "1 = Lung Cancer Detected" if prediction == 1 else "0 = No Lung Cancer"

            return render_template("predict.html", prediction=result)

        except Exception as e:
            flash(f"Error: {e}", "danger")
            return redirect(url_for('predict'))

    return render_template("predict.html", prediction=None)

# ============================
# 4. Charts Page
# ============================
@app.route('/charts')
def charts():
    # Example: simple bar chart of sample data
    data = {"Cancer": 60, "No Cancer": 40}
    labels = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['red', 'green'])
    plt.title("Sample Distribution of Predictions")
    plt.ylabel("Count")

    chart_path = "static/chart.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("charts.html", chart_url=chart_path)

if __name__ == "__main__":
    app.run(debug=True)
