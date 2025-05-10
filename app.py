from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("cleaned_medical_dataset.csv")

# Ensure the "name" column is of string type
df["name"] = df["name"].astype(str)

# Load model and encoders
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoders.pkl", "rb") as le_file:
    label_encoders = pickle.load(le_file)

# Prediction logic
def get_disease(medicine_name):
    medicine_name = medicine_name.strip().lower()
    # Again, ensure data in 'name' is string before applying .str.lower()
    match = df[df["name"].astype(str).str.lower() == medicine_name]

    if not match.empty:
        encoded_disease = match["indication"].values[0]
        disease = label_encoders['indication'].inverse_transform([encoded_disease])[0]
        medicine_details = match.iloc[0].to_dict()
        medicine_details['disease'] = disease
        return medicine_details
    else:
        return None

# Flask route
@app.route("/", methods=["GET", "POST"])
def home():
    medicine_details = None
    if request.method == "POST":
        medicine_name = request.form.get("medicine_name", "").strip()
        if medicine_name:
            medicine_details = get_disease(medicine_name)
    return render_template("index.html", medicine_details=medicine_details)

if __name__ == "__main__":
    app.run(debug=True)
