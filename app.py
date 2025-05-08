from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the Excel dataset
df = pd.read_excel("MEDICAL_DATASET.xlsx")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        medicine = request.form['medicine'].strip().lower()
        match = df[df['Name'].str.lower() == medicine]
        if not match.empty:
            result = match.iloc[0]['Indication']
        else:
            result = "Medicine not found in the dataset."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
