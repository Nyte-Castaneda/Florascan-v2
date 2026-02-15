#!/usr/bin/env python3
"""
API Test Script for FloraScan
Tests all major API endpoints
"""

import requests
import json

BASE_URL = 'http://localhost:5000/api'

def test_get_plants():
    """Test GET /api/plants"""
    print("\n1. Testing GET /api/plants...")
    response = requests.get(f'{BASE_URL}/plants')
    if response.status_code == 200:
        plants = response.json()
        print(f"   ✓ Success! Found {len(plants)} plants")
        if plants:
            print(f"   First plant: {plants[0]['common_name']}")
    else:
        print(f"   ✗ Failed with status {response.status_code}")

def test_search_plants():
    """Test GET /api/plants with search"""
    print("\n2. Testing plant search...")
    response = requests.get(f'{BASE_URL}/plants?search=Spider')
    if response.status_code == 200:
        plants = response.json()
        print(f"   ✓ Success! Found {len(plants)} matching plants")
    else:
        print(f"   ✗ Failed with status {response.status_code}")

def test_native_plants():
    """Test GET /api/plants?native=true"""
    print("\n3. Testing native plants filter...")
    response = requests.get(f'{BASE_URL}/plants?native=true')
    if response.status_code == 200:
        plants = response.json()
        print(f"   ✓ Success! Found {len(plants)} native Philippine plants")
        for plant in plants[:3]:
            print(f"      - {plant['common_name']}")
    else:
        print(f"   ✗ Failed with status {response.status_code}")

def test_get_plant_by_id():
    """Test GET /api/plants/<id>"""
    print("\n4. Testing GET plant by ID...")
    response = requests.get(f'{BASE_URL}/plants/1')
    if response.status_code == 200:
        plant = response.json()
        print(f"   ✓ Success! Got plant: {plant['common_name']}")
    else:
        print(f"   ✗ Failed with status {response.status_code}")

def test_identify_plant_text():
    """Test POST /api/identify with text description"""
    print("\n5. Testing plant identification (text only)...")
    data = {'description': 'spider plant with striped leaves'}
    response = requests.post(f'{BASE_URL}/identify', data=data)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Success! Identified as: {result['plant']['common_name']}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
        return result['request_id']
    else:
        print(f"   ✗ Failed with status {response.status_code}")
        return None

def test_submit_feedback(request_id):
    """Test POST /api/feedback"""
    if not request_id:
        print("\n6. Skipping feedback test (no request_id)")
        return
    
    print("\n6. Testing feedback submission...")
    data = {
        'request_id': request_id,
        'rating': 1,
        'comment': 'Test feedback'
    }
    response = requests.post(
        f'{BASE_URL}/feedback',
        json=data,
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code == 200:
        print(f"   ✓ Success! Feedback submitted")
    else:
        print(f"   ✗ Failed with status {response.status_code}")

def test_get_stats():
    """Test GET /api/stats"""
    print("\n7. Testing GET /api/stats...")
    response = requests.get(f'{BASE_URL}/stats')
    if response.status_code == 200:
        stats = response.json()
        print(f"   ✓ Success! Stats retrieved:")
        print(f"      Total plants: {stats['total_plants']}")
        print(f"      Native plants: {stats['native_plants']}")
        print(f"      Total identifications: {stats['total_identifications']}")
    else:
        print(f"   ✗ Failed with status {response.status_code}")

def main():
    print("="*60)
    print("FloraScan API Test Suite")
    print("="*60)
    print("\nMake sure the Flask server is running on port 5000!")
    print("Run: python app.py")
    
    try:
        # Test endpoints
        test_get_plants()
        test_search_plants()
        test_native_plants()
        test_get_plant_by_id()
        request_id = test_identify_plant_text()
        test_submit_feedback(request_id)
        test_get_stats()
        
        print("\n" + "="*60)
        print("✓ All tests completed!")
        print("="*60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to server!")
        print("Make sure the Flask server is running:")
        print("  python app.py\n")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}\n")

if __name__ == "__main__":
    main()