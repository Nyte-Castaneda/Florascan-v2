print("APP.PY LOADED ✅")
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess
try:
    from plant_inference_torch import predict_image as torch_predict_image
except:
    torch_predict_image = None

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
DATABASE = 'florascan.db'

# Project folder (where app.py lives)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try these in order (first one that loads wins): extracted folder, then .keras, then .h5
MODEL_CANDIDATES = [
    os.path.join(PROJECT_DIR, 'plant_model.keras'),
    os.path.join(PROJECT_DIR, 'plant_model_extracted'),  # extracted from plant_model.keras
]

# Class labels: exact order from training (cell 8FtVS2h8BR9z)
CLASS_LABELS = [
    'JADE PLANT',
    'OTHER',
    'PANDAKAKI',
    'SNAKE PLANT',
    'SPIDER PLANT',
    'TI PLANT',
]
# Map model output names to database common_name for lookup
MODEL_TO_DB_NAME = {
    'JADE PLANT': 'Jade Plant',
    'PANDAKAKI': 'Pandakaki',
    'SNAKE PLANT': 'Snake Plant',
    'SPIDER PLANT': 'Spider Plant',
    'TI PLANT': 'Ti Plant',
    'OTHER': None
}

plant_model = None
MODEL_PATH = None
for path in MODEL_CANDIDATES:
    if not os.path.exists(path):
        continue
    try:
        plant_model = load_model(path)
        MODEL_PATH = path
        print(f"✓ Loaded plant model from {MODEL_PATH}")

        print("\n===== MODEL DEBUG START =====")
        print("Loaded model file:", MODEL_PATH)
        print("MODEL INPUT SHAPE:", plant_model.input_shape)
        print("MODEL OUTPUT SHAPE:", plant_model.output_shape)
        print("LAST LAYER:", plant_model.layers[-1].name)
        try:
            print("LAST LAYER OUTPUT:", plant_model.layers[-1].output_shape)
        except Exception:
            pass
        print("===== MODEL DEBUG END =====\n")


        print("✓ Photo identification will use your trained dataset (AI model).")
        # Log input shape so preprocessing matches
        try:
            shp = plant_model.input.shape
            if shp.rank >= 3:
                print(f"  Model input shape: (height={int(shp[1])}, width={int(shp[2])}, channels={int(shp[3])})")
        except Exception:
            pass
        break
    except Exception as e:
        print(f"⚠ Could not load {path}: {e}")
if plant_model is None:
    print("⚠ No valid model found. Photo identification will use fallback (random/text).")
    print("  To use your trained model: copy 'plant_model (2).keras' into this folder and rename it to 'plant_model.keras'")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_db():
    """Get database connection"""
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initialize the database with tables"""
    db = get_db()
    cursor = db.cursor()
    
    # Plants table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            common_name TEXT NOT NULL,
            scientific_name TEXT NOT NULL,
            family TEXT,
            description TEXT,
            native_to_philippines BOOLEAN DEFAULT 0,
            care_instructions TEXT,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Identification requests table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS identification_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            description TEXT,
            identified_plant_id INTEGER,
            confidence_score REAL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (identified_plant_id) REFERENCES plants (id)
        )
    ''')
    
    # User feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER,
            rating INTEGER,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (request_id) REFERENCES identification_requests (id)
        )
    ''')
    
    # Search history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_term TEXT NOT NULL,
            result_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    db.commit()
    db.close()

def seed_data():
    """Populate database with initial plant data"""
    db = get_db()
    cursor = db.cursor()
    
    # Check if data already exists
    cursor.execute('SELECT COUNT(*) as count FROM plants')
    if cursor.fetchone()['count'] > 0:
        db.close()
        return
    
    plants = [
        {
            'common_name': 'Spider Plant',
            'scientific_name': 'Chlorophytum comosum',
            'family': 'Asparagaceae',
            'description': 'A popular houseplant with long, arching leaves that are striped with white or yellow. Easy to grow and propagate.',
            'native_to_philippines': 0,
            'care_instructions': 'Prefers bright, indirect light. Water when soil is dry. Tolerates a wide range of conditions.',
            'image_url': 'https://images.unsplash.com/photo-1638842940758-c933b1d7a369'
        },
        {
            'common_name': 'Snake Plant',
            'scientific_name': 'Sansevieria trifasciata',
            'family': 'Asparagaceae',
            'description': 'A hardy succulent with upright, sword-like leaves. Known for air-purifying qualities.',
            'native_to_philippines': 0,
            'care_instructions': 'Very low maintenance. Can tolerate low light and irregular watering. Avoid overwatering.',
            'image_url': 'https://images.unsplash.com/photo-1596670121720-43b064147efd'
        },
        {
            'common_name': 'Jade Plant',
            'scientific_name': 'Crassula ovata',
            'family': 'Crassulaceae',
            'description': 'A succulent with thick, fleshy leaves. Often called the "money plant" and believed to bring good fortune.',
            'native_to_philippines': 0,
            'care_instructions': 'Needs bright light. Water sparingly, allowing soil to dry between waterings.',
            'image_url': 'https://images.unsplash.com/photo-1757553774203-43c4a6e9ec47'
        },
        {
            'common_name': 'Pandakaki',
            'scientific_name': 'Tabernaemontana pandacaqui',
            'family': 'Apocynaceae',
            'description': 'A native Philippine shrub with fragrant white flowers. Used in traditional medicine.',
            'native_to_philippines': 1,
            'care_instructions': 'Thrives in tropical climates. Needs regular watering and partial shade to full sun.',
            'image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/03253jfDwarf_Tabernaemontana_pandacaqui_Philippinesfvf_01.jpg/800px-03253jfDwarf_Tabernaemontana_pandacaqui_Philippinesfvf_01.jpg'
        },
        {
            'common_name': 'Ti Plant',
            'scientific_name': 'Cordyline fruticosa',
            'family': 'Asparagaceae',
            'description': 'A tropical plant with colorful, broad leaves. Popular in landscaping across the Philippines.',
            'native_to_philippines': 0,
            'care_instructions': 'Prefers bright, indirect light and high humidity. Keep soil consistently moist.',
            'image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ti_plant_%28Cordyline_fruticosa%29.jpg/800px-Ti_plant_%28Cordyline_fruticosa%29.jpg'
        }
    ]
    
    for plant in plants:
        cursor.execute('''
            INSERT INTO plants (common_name, scientific_name, family, description, 
                              native_to_philippines, care_instructions, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (plant['common_name'], plant['scientific_name'], plant['family'],
              plant['description'], plant['native_to_philippines'],
              plant['care_instructions'], plant['image_url']))
    
    db.commit()
    db.close()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model_input_size():
    """Get (height, width) the loaded model expects. Default (224, 224) if unknown."""
    if plant_model is None:
        return (224, 224)
    try:
        shape = plant_model.input.shape
        if shape.rank >= 3:
            # Typical: (None, height, width, channels)
            h, w = int(shape[1]), int(shape[2])
            if h > 0 and w > 0:
                return (h, w)
    except Exception:
        pass
    return (224, 224)


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess for MobileNetV2: resize 224x224, then mobilenet_v2.preprocess_input.
    IMPORTANT: preprocess_input expects pixel values in 0..255 (NOT already divided by 255).
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)

    img_array = np.array(img).astype('float32')   # ✅ keep 0..255
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_plant_from_image(image_path):
    """
    Use the trained model to predict the plant label and confidence.
    Returns (label, confidence, all_probs) or (None, 0.0, []) if prediction fails.
    all_probs is a list of (class_label, probability) sorted by probability descending.
    """
    if plant_model is None:
        return None, 0.0, []

    try:
        img_array = preprocess_image(image_path)
        # Quick check: different images should have different fingerprints
        inp_mean, inp_std = float(np.mean(img_array)), float(np.std(img_array))
        print(f"  Input fingerprint: mean={inp_mean:.4f} std={inp_std:.4f} (should vary per image)")
        preds = plant_model.predict(img_array)
        if preds is None or len(preds) == 0:
            return None, 0.0, []

        probs = preds[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        # Log full prediction so we can see if different images give different results
        prob_str = ", ".join(f"{CLASS_LABELS[i]}: {probs[i]:.3f}" for i in range(min(len(CLASS_LABELS), len(probs))))
        print(f"  All classes: {prob_str}")

        if idx < 0 or idx >= len(CLASS_LABELS):
            return None, 0.0, []

        label = CLASS_LABELS[idx]
        # Build list of (label, prob) for all classes, sorted by prob descending (for alternatives)
        all_probs = [(CLASS_LABELS[i], float(probs[i])) for i in range(min(len(CLASS_LABELS), len(probs)))]
        all_probs.sort(key=lambda x: -x[1])
        return label, confidence, all_probs
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None, 0.0, []

# ============ Web Routes (Template Rendering) ============

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """Serve the about page"""
    return render_template('about.html')

# ============ API Routes ============

@app.route('/api/plants', methods=['GET'])
def get_plants():
    """Get all plants or filter by search term"""
    search_term = request.args.get('search', '')
    native_only = request.args.get('native', '').lower() == 'true'
    
    db = get_db()
    cursor = db.cursor()
    
    query = 'SELECT * FROM plants WHERE 1=1'
    params = []
    
    if search_term:
        query += ' AND (common_name LIKE ? OR scientific_name LIKE ? OR description LIKE ?)'
        search_pattern = f'%{search_term}%'
        params.extend([search_pattern, search_pattern, search_pattern])
    
    if native_only:
        query += ' AND native_to_philippines = 1'
    
    query += ' ORDER BY common_name'
    
    cursor.execute(query, params)
    plants = [dict(row) for row in cursor.fetchall()]
    
    # Log search if there was a search term
    if search_term:
        cursor.execute('''
            INSERT INTO search_history (search_term, result_count)
            VALUES (?, ?)
        ''', (search_term, len(plants)))
        db.commit()
    
    db.close()
    return jsonify(plants)

@app.route('/api/plants/<int:plant_id>', methods=['GET'])
def get_plant(plant_id):
    """Get a specific plant by ID"""
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('SELECT * FROM plants WHERE id = ?', (plant_id,))
    plant = cursor.fetchone()
    
    db.close()
    
    if plant:
        return jsonify(dict(plant))
    return jsonify({'error': 'Plant not found'}), 404

@app.route('/api/identify', methods=['POST'])
def identify_plant():
    """Handle plant identification request"""
    image_path = None
    description = request.form.get('description', '')

    # Handle file upload
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_path = filepath

    if not image_path and not description:
        return jsonify({'error': 'Please provide an image or description'}), 400

    db = get_db()
    cursor = db.cursor()

    identified_plant_id = None
    confidence_score = 0.0
    predicted_label = None
    model_confidence = 0.0
    all_probs = []  # list of (label, prob)

    # 1) Image-based prediction (Torch -> Keras fallback)
    if image_path:
        if torch_predict_image:
            predicted_label, model_confidence, all_probs = torch_predict_image(image_path)
        else:
            predicted_label = None
            model_confidence = 0.0
            all_probs = []

        print(f"Model prediction (torch): label={predicted_label}, confidence={model_confidence}")

        # If torch didn't give a label, try keras
        if not predicted_label:
            predicted_label, model_confidence, all_probs = predict_plant_from_image(image_path)
            print(f"Model prediction (keras): label={predicted_label}, confidence={model_confidence}")

        # If model predicts OTHER, do not map to DB / do not return plant info
        if predicted_label == "OTHER":
            confidence_score = float(model_confidence or 0.0)

        # If it's one of the 5 plants, map to DB
        elif predicted_label:
            db_name = MODEL_TO_DB_NAME.get(predicted_label, predicted_label)
            cursor.execute('SELECT id FROM plants WHERE common_name = ? LIMIT 1', (db_name,))
            result = cursor.fetchone()
            if result:
                identified_plant_id = result['id']
                confidence_score = float(model_confidence or 0.0)

    # 2) If still not identified and text description exists, try text search
    if identified_plant_id is None and description:
        search_pattern = f'%{description}%'
        cursor.execute('''
            SELECT id FROM plants
            WHERE common_name LIKE ? OR scientific_name LIKE ? OR description LIKE ?
            LIMIT 1
        ''', (search_pattern, search_pattern, search_pattern))

        result = cursor.fetchone()
        if result:
            identified_plant_id = result['id']
            confidence_score = max(confidence_score, 0.75)

    # Store identification request
    cursor.execute('''
        INSERT INTO identification_requests
        (image_path, description, identified_plant_id, confidence_score, status)
        VALUES (?, ?, ?, ?, ?)
    ''', (image_path, description, identified_plant_id, confidence_score, 'completed'))
    request_id = cursor.lastrowid

    # Get identified plant details (if we matched a DB entry)
    plant = None
    if identified_plant_id is not None:
        cursor.execute('SELECT * FROM plants WHERE id = ?', (identified_plant_id,))
        plant = cursor.fetchone()

    db.commit()
    db.close()

    # Build response payload
    if plant:
        plant_payload = dict(plant)

    elif predicted_label == "OTHER":
        plant_payload = None  # ✅ key behavior: OTHER returns no plant info

    elif predicted_label:
        # If predicted label isn't in DB (should be rare), return label only
        plant_payload = {
            'common_name': MODEL_TO_DB_NAME.get(predicted_label, predicted_label),
            'scientific_name': '',
            'family': '',
            'description': '',
            'native_to_philippines': 0,
            'care_instructions': '',
            'image_url': ''
        }
        confidence_score = max(confidence_score, float(model_confidence or 0.0))

    else:
        plant_payload = None

    # Alternatives (only if low confidence and we have probs)
    alternatives = []
    main_name = plant_payload.get('common_name', '') if plant_payload else ''
    if confidence_score < 0.5 and all_probs:
        if not main_name and all_probs:
            main_name = MODEL_TO_DB_NAME.get(all_probs[0][0], all_probs[0][0])
        for label, prob in all_probs:
            if prob > 0.01:
                display_name = MODEL_TO_DB_NAME.get(label, label)
                if display_name != main_name and display_name is not None:
                    alternatives.append({'name': display_name, 'confidence': round(prob, 4)})
            if len(alternatives) >= 4:
                break

    # Invalid result logic
    INVALID_CONFIDENCE_THRESHOLD = 0.60

    # If OTHER -> always treat as invalid identification (but not an error)
    if predicted_label == "OTHER":
        invalid_result = True
    else:
        invalid_result = (plant_payload is not None and confidence_score < INVALID_CONFIDENCE_THRESHOLD)

    if invalid_result:
        # do not show plant card when invalid
        if predicted_label != "OTHER":
            plant_payload = None
            identified_plant_id = None

    response = {
        'request_id': request_id,
        'plant': plant_payload,
        'confidence': float(confidence_score or 0.0),
        'status': 'completed',
        'low_confidence': confidence_score < 0.5 and plant_payload is not None,
        'invalid_result': invalid_result,
        'alternatives': alternatives,
    }

    # Optional message for OTHER
    if predicted_label == "OTHER":
        response['message'] = 'Not a supported plant (OTHER).'

    return jsonify(response)


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for an identification"""
    data = request.get_json()
    
    request_id = data.get('request_id')
    rating = data.get('rating')
    comment = data.get('comment', '')
    
    if not request_id or rating is None:
        return jsonify({'error': 'Missing required fields'}), 400
    
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('''
        INSERT INTO feedback (request_id, rating, comment)
        VALUES (?, ?, ?)
    ''', (request_id, rating, comment))
    
    db.commit()
    db.close()
    
    return jsonify({'message': 'Feedback submitted successfully'})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get application statistics"""
    db = get_db()
    cursor = db.cursor()
    
    # Total plants
    cursor.execute('SELECT COUNT(*) as total FROM plants')
    total_plants = cursor.fetchone()['total']
    
    # Native plants
    cursor.execute('SELECT COUNT(*) as total FROM plants WHERE native_to_philippines = 1')
    native_plants = cursor.fetchone()['total']
    
    # Total identifications
    cursor.execute('SELECT COUNT(*) as total FROM identification_requests')
    total_identifications = cursor.fetchone()['total']
    
    # Average confidence
    cursor.execute('SELECT AVG(confidence_score) as avg FROM identification_requests')
    avg_confidence = cursor.fetchone()['avg'] or 0
    
    db.close()
    
    return jsonify({
        'total_plants': total_plants,
        'native_plants': native_plants,
        'total_identifications': total_identifications,
        'average_confidence': round(avg_confidence, 2)
    })

if __name__ == '__main__':
    print("="*60)
    print("FloraScan - Plant Identification System")
    print("="*60)
    init_db()
    seed_data()
    print("✓ Database initialized and seeded!")
    print("✓ Starting Flask server...")
    print("="*60)
    print("\nAccess the application at:")
    print("  → http://localhost:5000")
    print("  → http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)