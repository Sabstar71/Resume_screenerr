from flask import Flask, request, jsonify, render_template
import os
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util

# Init app and model
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & accurate for semantic similarity

# Load job description
with open("job_description.txt", "r", encoding='utf-8') as file:
    job_description = file.read()

# Extract text from PDF resume
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# Semantic similarity score
def compute_similarity(text1, text2):
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(score * 100, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        resume_text = extract_text_from_pdf(filepath)
        match_score = compute_similarity(job_description, resume_text)

        return jsonify({
            'filename': filename,
            'match_score': match_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
