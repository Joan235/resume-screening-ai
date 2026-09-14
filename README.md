# AI Resume Screening System

Automated Resume Classification with Flask and Machine Learning

A portfolio-ready web application that classifies uploaded resumes into job categories using a TF-IDF feature extraction pipeline and a scikit-learn prediction model.

## Business Problem

Hiring teams receive a high volume of resumes for each job opening, making manual screening slow, repetitive, and difficult to scale. This project addresses that challenge with a lightweight AI-assisted screening workflow that can quickly categorize resumes and help recruiters prioritize applications.

## Project Overview

This project provides a lightweight prototype for automated resume classification. Users upload a resume, the application extracts and processes the text, and the model predicts the job category with a confidence score.

The project combines:

- A Flask web interface for uploading resumes
- Text extraction from PDF and text files using PyPDF2
- A scikit-learn text classification model powered by TF-IDF features
- A clean responsive frontend using Tailwind CSS

## Features

- Upload resume files in PDF or text format
- Extract text from PDF resumes automatically
- Convert raw resume text into TF-IDF features
- Predict the probable job category for the uploaded resume
- Display confidence score alongside the predicted category
- Show a warning message when the uploaded file appears to be low confidence or not a valid resume-like document
- Simple, attractive web UI built with Flask templates and Tailwind CSS

## Tech Stack

- Python
- Flask
- scikit-learn
- joblib
- PyPDF2
- HTML, CSS, Tailwind CSS

## How It Works

1. The user uploads a resume as a PDF or text file.
2. The Flask app reads the uploaded file.
3. If the file is a PDF, text is extracted page by page using PyPDF2.
4. The extracted resume text is transformed into a TF-IDF vector representation.
5. The trained model predicts the job category associated with the resume.
6. The app returns the prediction and confidence percentage through the frontend UI.

## Future Improvements

- Add job role ranking and candidate shortlisting scores
- Improve model accuracy with more labeled resume data
- Add resume parsing for structured fields such as skills, degree, experience, and location
- Add authentication and user management for recruiters
- Replace the single-page upload design with a more complete dashboard experience

## License

This project is open for educational and portfolio demonstration purposes.

## Author

Built as an AI-powered resume screening prototype for learning, experimentation, and portfolio presentation.
