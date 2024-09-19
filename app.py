from flask import Flask, request, send_from_directory, jsonify, session
# from flask_session import Session
from werkzeug.utils import secure_filename
import os
import uuid
from rag_pipeline import RAGPipeline

app = Flask(__name__, static_url_path='', static_folder='static')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

rag_pipeline = RAGPipeline()
RESPONSES = []

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            print(f"File saved to: {filepath}")
            
            # Process the PDF using the RAG pipeline
            rag_pipeline.process_document(filepath)
            
            return jsonify({
                'message': 'File uploaded and processed successfully'
            }), 200
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # chat_history = responses[-1]['chat_history'] if responses else None
        print(len(RESPONSES))
        response = rag_pipeline.get_answer(question, chat_history=RESPONSES[-1]['chat_history'] if RESPONSES else None)
        # answer_str = response.get('answer', str(response))
        RESPONSES.append(response)
        # print(chat_history)
        
        return jsonify({
            'answer': str(response['answer'])
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)