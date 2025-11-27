from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# -----------------------------
# Load Model + Encoders
# -----------------------------
model = pickle.load(open("rf-model-local.pkl", "rb"))

encoders = {}
cat_cols = ["SUPPLIER", "ITEM TYPE"]   # update if more encoders exist

for col in cat_cols:
    encoders[col] = pickle.load(open(f"{col}_encoder.pkl", "rb"))

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def index():
    return render_template("index2.html")

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Read user input
    YEAR = int(request.form["year"])
    MONTH = int(request.form["month"])
    SUPPLIER = request.form["supplier"]
    ITEM_TYPE = request.form["item_type"]
    RETAIL_TRANSFERS = float(request.form["retail_transfers"])
    WAREHOUSE_SALES = float(request.form["warehouse_sales"])

    # Create dataframe
    input_df = pd.DataFrame([{
        "YEAR": YEAR,
        "MONTH": MONTH,
        "SUPPLIER": SUPPLIER,
        "ITEM TYPE": ITEM_TYPE,
        "RETAIL TRANSFERS": RETAIL_TRANSFERS,
        "WAREHOUSE SALES": WAREHOUSE_SALES
    }])

    # Fill unknowns and encode
    input_df.fillna("Unknown", inplace=True)

    for col in cat_cols:
        input_df[col] = encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]

    return render_template("index2.html",
                           predicted_sales=round(prediction, 2))


if __name__ == "__main__":
    app.run(debug=True)
