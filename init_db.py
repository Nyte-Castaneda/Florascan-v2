
"""
Standalone Database Initialization Script
Run this to create and seed the FloraScan database without starting the Flask server
"""

import sqlite3
from datetime import datetime

DATABASE = 'florascan.db'

def init_db():
    """Initialize the database with tables"""
    print("Initializing database...")
    db = sqlite3.connect(DATABASE)
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
    print("✓ Database tables created successfully!")

def seed_data():
    """Populate database with initial plant data"""
    print("\nSeeding database with plant data...")
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    
    # Check if data already exists
    cursor.execute('SELECT COUNT(*) as count FROM plants')
    if cursor.fetchone()[0] > 0:
        print("✓ Database already contains plant data. Skipping seed.")
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
    print(f"✓ Successfully added {len(plants)} plants to database!")

def main():
    print("="*60)
    print("FloraScan Database Initialization")
    print("="*60)
    
    try:
        init_db()
        seed_data()
        
        print("\n" + "="*60)
        print("Database initialization complete!")
        print(f"Database file: {DATABASE}")
        print("="*60)
        print("\nYou can now:")
        print("1. Run 'python app.py' to start the Flask server")
        print("2. Run 'python view_database.py' to view database contents")
        print("3. Run 'python test_api.py' to test the API endpoints")
        print()
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")

if __name__ == "__main__":
    main()