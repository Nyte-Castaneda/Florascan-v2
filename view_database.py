#!/usr/bin/env python3
"""
Database Viewer for FloraScan
Simple CLI tool to view database contents
"""

import sqlite3
from datetime import datetime

DATABASE = 'florascan.db'

def print_table(title, headers, rows):
    """Pretty print a database table"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    if not rows:
        print("No data found.")
        return
    
    # Print headers
    header_line = " | ".join(str(h).ljust(15)[:15] for h in headers)
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in rows:
        row_line = " | ".join(str(v).ljust(15)[:15] if v is not None else "NULL".ljust(15) for v in row)
        print(row_line)
    
    print(f"\nTotal records: {len(rows)}")

def view_plants():
    """View all plants in the database"""
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    
    cursor.execute('SELECT id, common_name, scientific_name, family, native_to_philippines FROM plants ORDER BY common_name')
    rows = cursor.fetchall()
    
    print_table(
        "PLANTS TABLE",
        ["ID", "Common Name", "Scientific Name", "Family", "Native to PH"],
        rows
    )
    
    db.close()

def view_identification_requests():
    """View recent identification requests"""
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    
    cursor.execute('''
        SELECT 
            r.id,
            SUBSTR(r.description, 1, 20) as description,
            p.common_name,
            ROUND(r.confidence_score * 100, 1) as confidence,
            r.status,
            r.created_at
        FROM identification_requests r
        LEFT JOIN plants p ON r.identified_plant_id = p.id
        ORDER BY r.created_at DESC
        LIMIT 20
    ''')
    rows = cursor.fetchall()
    
    print_table(
        "RECENT IDENTIFICATION REQUESTS",
        ["ID", "Description", "Identified As", "Confidence%", "Status", "Created At"],
        rows
    )
    
    db.close()

def view_feedback():
    """View user feedback"""
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    
    cursor.execute('''
        SELECT 
            f.id,
            f.request_id,
            f.rating,
            SUBSTR(f.comment, 1, 30) as comment,
            f.created_at
        FROM feedback f
        ORDER BY f.created_at DESC
        LIMIT 20
    ''')
    rows = cursor.fetchall()
    
    print_table(
        "USER FEEDBACK",
        ["ID", "Request ID", "Rating", "Comment", "Created At"],
        rows
    )
    
    db.close()

def view_search_history():
    """View search history"""
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    
    cursor.execute('''
        SELECT 
            search_term,
            COUNT(*) as search_count,
            AVG(result_count) as avg_results,
            MAX(created_at) as last_search
        FROM search_history
        GROUP BY search_term
        ORDER BY search_count DESC
        LIMIT 15
    ''')
    rows = cursor.fetchall()
    
    print_table(
        "POPULAR SEARCHES",
        ["Search Term", "Times Searched", "Avg Results", "Last Search"],
        rows
    )
    
    db.close()

def view_statistics():
    """View overall statistics"""
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    
    # Total plants
    cursor.execute('SELECT COUNT(*) FROM plants')
    total_plants = cursor.fetchone()[0]
    
    # Native plants
    cursor.execute('SELECT COUNT(*) FROM plants WHERE native_to_philippines = 1')
    native_plants = cursor.fetchone()[0]
    
    # Total identifications
    cursor.execute('SELECT COUNT(*) FROM identification_requests')
    total_identifications = cursor.fetchone()[0]
    
    # Average confidence
    cursor.execute('SELECT AVG(confidence_score) FROM identification_requests WHERE confidence_score > 0')
    avg_confidence = cursor.fetchone()[0] or 0
    
    # Total feedback
    cursor.execute('SELECT COUNT(*) FROM feedback')
    total_feedback = cursor.fetchone()[0]
    
    # Average rating
    cursor.execute('SELECT AVG(rating) FROM feedback')
    avg_rating = cursor.fetchone()[0] or 0
    
    # Positive feedback percentage
    cursor.execute('SELECT COUNT(*) FROM feedback WHERE rating = 1')
    positive_feedback = cursor.fetchone()[0]
    feedback_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
    
    print(f"\n{'='*80}")
    print("DATABASE STATISTICS")
    print(f"{'='*80}")
    print(f"Total Plants:               {total_plants}")
    print(f"Native to Philippines:      {native_plants} ({native_plants/total_plants*100:.1f}%)")
    print(f"Total Identifications:      {total_identifications}")
    print(f"Average Confidence:         {avg_confidence*100:.1f}%")
    print(f"Total Feedback Received:    {total_feedback}")
    print(f"Average Rating:             {avg_rating:.2f}/1.00")
    print(f"Positive Feedback Rate:     {feedback_rate:.1f}%")
    print(f"{'='*80}\n")
    
    db.close()

def show_menu():
    """Display main menu"""
    print("\n" + "="*80)
    print("FloraScan Database Viewer")
    print("="*80)
    print("\n1. View Plants")
    print("2. View Identification Requests")
    print("3. View Feedback")
    print("4. View Search History")
    print("5. View Statistics")
    print("6. View All")
    print("0. Exit")
    print("\nEnter your choice: ", end="")

def main():
    """Main program loop"""
    try:
        db = sqlite3.connect(DATABASE)
        db.close()
    except:
        print(f"\n❌ ERROR: Database '{DATABASE}' not found!")
        print("Please run 'python app.py' first to initialize the database.\n")
        return
    
    while True:
        show_menu()
        choice = input().strip()
        
        if choice == "1":
            view_plants()
        elif choice == "2":
            view_identification_requests()
        elif choice == "3":
            view_feedback()
        elif choice == "4":
            view_search_history()
        elif choice == "5":
            view_statistics()
        elif choice == "6":
            view_statistics()
            view_plants()
            view_identification_requests()
            view_feedback()
            view_search_history()
        elif choice == "0":
            print("\nGoodbye!\n")
            break
        else:
            print("\n❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()