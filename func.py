import requests
from bs4 import BeautifulSoup
import os
import re

BASE_URL = "https://www.who.int"
FACT_SHEETS_INDEX = "https://www.who.int/news-room/fact-sheets"

# Create output folder
output_folder = "who_fact_sheets_txt"
os.makedirs(output_folder, exist_ok=True)

# ---- Step 1: Fetch all topic links ----
print("Fetching main fact sheets index page...")
resp = requests.get(FACT_SHEETS_INDEX)
soup = BeautifulSoup(resp.text, "html.parser")

links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    # pattern: /news-room/fact-sheets/detail/<slug>
    if "/news-room/fact-sheets/detail/" in href:
        full_url = BASE_URL + href
        if full_url not in links:
            links.append(full_url)

print(f"Found {len(links)} topic links!")

# ---- Step 2: Visit each link & extract text ----
for idx, url in enumerate(links, start=1):
    print(f"[{idx}/{len(links)}] Scraping {url}")
    page = requests.get(url)
    page_soup = BeautifulSoup(page.text, "html.parser")

    # Extract main article content
    article = page_soup.find("main")
    if not article:
        print("  ❗ No article found, skipping.")
        continue

    # Extract title
    title_tag = article.find(["h1", "h2"])
    title = title_tag.get_text(strip=True) if title_tag else "WHO_FACT_SHEET"

    # Clean filename
    filename = re.sub(r"[^a-zA-Z0-9 _-]", "", title).replace(" ", "_") + ".txt"
    filepath = os.path.join(output_folder, filename)

    # Collate all relevant text
    lines = []
    for child in article.find_all(["h1","h2","h3","p","li"]):
        text = child.get_text(strip=True)
        if text:
            lines.append(text)

    # Save to TXT
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))

    print(f"  ✅ Saved: {filename}")

print("All done!")