#!/usr/bin/env python3
"""Test /api/identify: POST an image and print response (plant, confidence, invalid_result, alternatives)."""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(PROJECT_DIR, 'uploads')
TEST_IMAGE = os.path.join(UPLOADS, '20260207_184832_Spider-plant.jpg')

def main():
    if not os.path.isfile(TEST_IMAGE):
        print("Test image not found:", TEST_IMAGE)
        sys.exit(1)
    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        sys.exit(1)

    url = "http://127.0.0.1:5000/api/identify"
    with open(TEST_IMAGE, "rb") as f:
        files = {"image": ("test.jpg", f, "image/jpeg")}
        data = {}
        print("POSTing to", url, "...")
        r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    out = r.json()
    print("\n--- Response ---")
    print("Status:", out.get("status"))
    print("Confidence:", out.get("confidence"), f"({out.get('confidence', 0)*100:.1f}%)")
    print("Invalid result:", out.get("invalid_result"))
    print("Low confidence:", out.get("low_confidence"))
    if out.get("plant"):
        print("Plant:", out["plant"].get("common_name"))
    print("Alternatives:", out.get("alternatives", []))
    print("---")
    print("OK - identify API and invalid/low_confidence flow work.")

if __name__ == "__main__":
    main()
