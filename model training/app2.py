from flask import Flask, render_template, request
import pandas as pd
import pickle
from google import genai
import os
from datetime import datetime

app = Flask(__name__)

# -----------------------------
# Configure Gemini AI
# -----------------------------
GEMINI_API_KEY = "AIzaSyBAxovSQMG7WXySS-yuO-86dy1SCW38s6k"  # Replace with your actual API key
client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------
# Load Model + Encoders
# -----------------------------
model = pickle.load(open("rf-model-local.pkl", "rb"))

encoders = {}
cat_cols = ["SUPPLIER", "ITEM TYPE"]

for col in cat_cols:
    encoders[col] = pickle.load(open(f"{col}_encoder.pkl", "rb"))

# -----------------------------
# AI Analysis Function
# -----------------------------
def get_ai_analysis(year, month, supplier, item_type, retail_transfers, warehouse_sales, predicted_sales):
    """Get AI-powered insights using Gemini"""
    
    prompt = f"""
    As a retail analytics expert, analyze this liquor sales prediction and provide insights:
    
    PREDICTION CONTEXT:
    - Year: {year}
    - Month: {month}
    - Supplier: {supplier}
    - Item Type: {item_type}
    - Retail Transfers: {retail_transfers} cases
    - Warehouse Sales: {warehouse_sales} cases
    - Predicted Retail Sales: ${predicted_sales}
    
    DATASET CONTEXT:
    This is liquor sales data with fractional case quantities. Retail Sales represent customer purchases, 
    Retail Transfers are inter-store movements, and Warehouse Sales are warehouse-to-store shipments.
    
    ANALYSIS REQUEST:

    1. Suggest optimization strategies for inventory and sales
    2. Provide seasonal insights if applicable
    3. Comment on supplier performance expectations

    
    Format your response in para with max of 5 lines.
    """
    print("Sending prompt to Gemini AI...")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Using the correct model name
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Analysis temporarily unavailable. Error: {str(e)}"

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def index():
    return render_template("index3.html")

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
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

        # Handle unseen labels in encoding
        for col in cat_cols:
            le = encoders[col]
            # Handle unseen labels by mapping them to a default value
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else "Unknown")
            input_df[col] = le.transform(input_df[col])

        # Predict
        prediction = model.predict(input_df)[0]
        predicted_sales = round(prediction, 2)

        # Get AI Analysis
        ai_analysis = get_ai_analysis(
            YEAR, MONTH, SUPPLIER, ITEM_TYPE, 
            RETAIL_TRANSFERS, WAREHOUSE_SALES, 
            predicted_sales
        )

        return render_template("index3.html",
                            predicted_sales=predicted_sales,
                            ai_analysis=ai_analysis)

    except Exception as e:
        return render_template("index3.html",
                            error_message=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)