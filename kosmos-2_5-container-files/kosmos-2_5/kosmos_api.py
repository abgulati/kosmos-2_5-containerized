from flask import Flask, request, jsonify
import subprocess
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = str(uuid.uuid4()) + secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save the file temporarily
        file.save(filepath)

        # Get the task type from form data
        task = request.form.get('task', 'md')

        # Run inference
        cmd = f"python3 inference.py --do_{task} --image {filepath} --ckpt ckpt.pt"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Clean the temp image file
        os.remove(filepath)

        return jsonify({'output': result.stdout, 'error': result.stderr})

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=25000)