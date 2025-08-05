from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from src.predictor import EWastePredictor
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictor
model_path = os.path.join("models", "e_waste_model.h5")
predictor = None

if os.path.exists(model_path):
    predictor = EWastePredictor(model_path)
else:
    print("\u26a0\ufe0f Model not found. Please train the model first.")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = predictor.predict_image(filepath)
        os.remove(filepath)

        if 'error' in result:
            return jsonify(result), 500

        ewaste_type = result['ewaste_type']
        confidence = result['confidence']
        top_3 = result['top_3_predictions']

        category_icons = {
            'keyboards': '⌨️',
            'mouse': '🖱️',
            'battery': '🔋',
            'mobiles': '📱',
            'pcb': '🔌',
            'microwave': '📺'
        }

        return render_template(
            'result.html',
            ewaste_type=ewaste_type,
            confidence=confidence,
            top_3=top_3,
            category_icons=category_icons
        )

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

# DEBUGGING TEMPLATE PATH ISSUE
print("\n===== TEMPLATE DEBUGGING =====")
print("Current working directory:", os.getcwd())
print("Flask template folder:", app.template_folder or "default: ./templates")
template_path = os.path.join(os.getcwd(), 'templates')
print("Does templates folder exist?", os.path.isdir(template_path))
if os.path.isdir(template_path):
    print("Files in templates folder:", os.listdir(template_path))
print("================================\n")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
