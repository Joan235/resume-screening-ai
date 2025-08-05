


print("Script started")
print("Importing Flask...")
from flask import Flask, render_template, request
print("Importing joblib...")
import joblib
print("Importing PyPDF2...")
import PyPDF2
print("Importing os...")
import os

try:
    print("Loading model...")
    model = joblib.load("resume_model.pkl")
    print("Model loaded.")
    print("Loading tfidf vectorizer...")
    tfidf = joblib.load("tfidf.pkl")
    print("TFIDF loaded.")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    with open("error.log", "w") as f:
        f.write(str(e))
    raise



app = Flask(__name__)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    print("Index route accessed")
    prediction = None
    warning = None
    if request.method == "POST":
        print("POST request received")
        uploaded_file = request.files['resume']
        if uploaded_file.filename.endswith(".pdf"):
            print("PDF uploaded")
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            print("Text file uploaded")
            resume_text = uploaded_file.read().decode("utf-8")

        # Transform text
        transformed_text = tfidf.transform([resume_text])
        pred = model.predict(transformed_text)[0]
        pred_proba = model.predict_proba(transformed_text).max() * 100

        warning = None
        if pred_proba < 10:
            warning = "Warning: This file does not appear to be a resume. Please upload a valid resume."
        prediction = f"Predicted Job Category: {pred} ({pred_proba:.2f}% confidence)"
    else:
        print("GET request received")

    return render_template("index.html", prediction=prediction, warning=warning)

# Simple test route to check if Flask is working
@app.route("/test")
def test():
    return "Flask is working!"

if __name__ == "__main__":
    print("About to start Flask app...")
    app.run(debug=True)
