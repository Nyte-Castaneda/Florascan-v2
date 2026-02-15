import os, time, random, re
import requests

OUT_DIR = "raw_other_plants"
os.makedirs(OUT_DIR, exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"

# Better categories than Houseplants (still not perfect, but much cleaner)
CATEGORIES = [
    "Category:Potted plants",
    "Category:Indoor plants",
    "Category:Houseplants",
]

TARGET = 120          # download extra; you only need 50-100
THUMB_WIDTH = 512     # IMPORTANT: thumbnail to avoid 429
SLEEP_API = 0.6       # slow down API calls
SLEEP_DL = 0.6        # slow down downloads

session = requests.Session()
session.headers.update({"User-Agent": "FloraScanDatasetBuilder/1.0 (educational project)"})

# quick filters to skip obvious non-plant files
BAD_WORDS = re.compile(r"(cat|dog|aircon|ac\b|window|laptop|phone|keyboard|unsplash|car|person|selfie)", re.I)

def get_category_files(category_title: str, limit: int = 800):
    titles = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmtype": "file",
            "cmlimit": 200,
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        r = session.get(API, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            title = m.get("title", "")
            if title.startswith("File:"):
                titles.append(title)

        if len(titles) >= limit:
            return titles[:limit]

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            return titles

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def get_thumb_urls(file_titles):
    """
    Get thumbnail URLs for many files at once using prop=imageinfo with iiurlwidth.
    This is MUCH lighter than downloading full-size originals.
    """
    # titles param can take many joined by |
    titles_param = "|".join(file_titles)
    params = {
        "action": "query",
        "prop": "imageinfo",
        "titles": titles_param,
        "iiprop": "url",
        "iiurlwidth": THUMB_WIDTH,
        "format": "json",
    }
    r = session.get(API, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    out = []
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        title = page.get("title")
        ii = page.get("imageinfo")
        if not title or not ii:
            continue
        thumburl = ii[0].get("thumburl")  # thumbnail link
        if thumburl:
            out.append((title, thumburl))
    return out

def safe_name(file_title: str):
    name = file_title.replace("File:", "")
    name = name.replace("/", "_")
    return name

def download(url: str, out_path: str):
    # retry with simple backoff on 429/5xx
    backoff = 1.0
    for attempt in range(6):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                if r.status_code == 429:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 20)
                    continue
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)
    return False

def main():
    # Gather titles from multiple categories
    titles = []
    for cat in CATEGORIES:
        print("Listing:", cat)
        titles.extend(get_category_files(cat, limit=600))
        time.sleep(SLEEP_API)

    # de-duplicate + shuffle
    titles = list(dict.fromkeys(titles))
    random.shuffle(titles)

    # filter by extension + quick bad word filter
    filtered = []
    for t in titles:
        fname = safe_name(t).lower()
        if not fname.endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        if BAD_WORDS.search(fname):
            continue
        filtered.append(t)

    print("Candidate files after filters:", len(filtered))

    downloaded = 0
    i = 0

    while downloaded < TARGET and i < len(filtered):
        batch = filtered[i:i+50]
        i += 50

        try:
            pairs = get_thumb_urls(batch)  # [(title, thumburl), ...]
        except Exception as e:
            print("API batch skip:", e)
            time.sleep(SLEEP_API * 2)
            continue

        time.sleep(SLEEP_API)

        for title, url in pairs:
            if downloaded >= TARGET:
                break

            fname = safe_name(title)
            out_path = os.path.join(OUT_DIR, fname)

            if os.path.exists(out_path):
                continue

            print(f"[{downloaded+1}/{TARGET}] {title} (thumb {THUMB_WIDTH}px)")
            ok = download(url, out_path)
            if ok:
                downloaded += 1
            else:
                print("  skip (download failed)")

            time.sleep(SLEEP_DL)

    print("\nDone.")
    print("Saved to:", os.path.abspath(OUT_DIR))
    print("Downloaded:", downloaded)

if __name__ == "__main__":
    main()