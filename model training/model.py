import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

print("Loading dataset...")

df = pd.read_csv("Warehouse_and_Retail_Sales.csv")

# Clean
df = df.dropna(subset=["RETAIL SALES"])
df.fillna("Unknown", inplace=True)

# Prepare features
X = df.drop(["RETAIL SALES", "ITEM DESCRIPTION", "ITEM CODE"], axis=1)
y = df["RETAIL SALES"]

# Encode categorical columns
cat_cols = X.select_dtypes(include="object").columns
encoders = {}

print("Encoding categorical columns...")

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
    pickle.dump(le, open(f"{col}_encoder.pkl", "wb"))

print("✓ Encoders saved.")

# Train Random Forest with sklearn 1.7.2
print("Training Random Forest with sklearn", __import__("sklearn").__version__)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# Save model
pickle.dump(model, open("rf-model-local.pkl", "wb"))
print("✓ Model saved as rf-model-local.pkl")
