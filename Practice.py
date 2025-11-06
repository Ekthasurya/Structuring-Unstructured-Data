# ======================================================
# 🌍 DSPy Practical Assignment - Structuring Unstructured Data 
# ======================================================

# --- 1️⃣ Install Dependencies ---
# !pip install dspy pydantic aiohttp beautifulsoup4 pandas tqdm

import dspy
from pydantic import BaseModel, Field
from typing import List
import asyncio, aiohttp, os
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()  # reads .env file




# --- 3️⃣ Configure DSPy LM ---
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# --- 4️⃣ URLs to Scrape ---
URLS = [
    "https://en.wikipedia.org/wiki/Sustainable_agriculture",
    "https://www.nature.com/articles/d41586-025-03353-5",
    "https://www.sciencedirect.com/science/article/pii/S1043661820315152",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10457221/",
    "https://www.fao.org/3/y4671e/y4671e06.htm",
    "https://www.medscape.com/viewarticle/time-reconsider-tramadol-chronic-pain-2025a1000ria",
    "https://www.sciencedirect.com/science/article/pii/S0378378220307088",
    "https://www.frontiersin.org/news/2025/09/01/rectangle-telescope-finding-habitable-planets",
    "https://www.medscape.com/viewarticle/second-dose-boosts-shingles-protection-adults-aged-65-years-2025a1000ro7",
    "https://www.theguardian.com/global-development/2025/oct/13/astro-ambassadors-stargazers-himalayas-hanle-ladakh-india",
]

# --- 5️⃣ Async Scraper ---
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

async def fetch_text(session, url):
    """Fetch webpage text asynchronously."""
    try:
        async with session.get(url, headers=HEADERS, timeout=30) as resp:
            if resp.status != 200:
                raise Exception(f"Status {resp.status}")
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all("p")]
            text = "\n".join(paragraphs)
            print(f"✅ Scraped {len(text)} chars from {url}")
            return url, text
    except Exception as e:
        print(f"⚠️ Error scraping {url}: {e}")
        return url, ""

async def scrape_all(urls):
    """Scrape all URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_text(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# --- 6️⃣ Entity Schema ---
class EntityWithAttr(BaseModel):
    entity: str = Field(description="Named entity in text")
    attr_type: str = Field(description="Type of entity (e.g. Concept, Crop, Drug, Process)")

# --- 7️⃣ DSPy Signature ---
class ExtractEntities(dspy.Signature):
    paragraph: str = dspy.InputField()
    entities: List[EntityWithAttr] = dspy.OutputField()

extract_entities = dspy.Predict(ExtractEntities)

# --- 8️⃣ Helpers ---
def deduplicate_entities(entity_list):
    seen, unique = set(), []
    for e in entity_list:
        key = e.entity.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique

def triples_to_mermaid(entities):
    """Generate a simple mermaid diagram from entities."""
    lines = ["graph TD"]
    for i in range(len(entities) - 1):
        src = entities[i].entity.replace(" ", "_")
        dst = entities[i + 1].entity.replace(" ", "_")
        lines.append(f"  {src} --> {dst}")
    return "\n".join(lines)

# --- 9️⃣ Main Processing ---
async def main():
    results = await scrape_all(URLS)
    all_data = []
    os.makedirs("outputs", exist_ok=True)

    for i, (url, text) in enumerate(results, start=1):
        if not text:
            continue

        sample_text = text[:2000]

        try:
            pred = extract_entities(paragraph=sample_text)
            entities = deduplicate_entities(pred.entities)
            print(f"✅ Extracted {len(entities)} unique entities from {url}")

            if entities:
                # Save mermaid diagram
                mermaid_data = triples_to_mermaid(entities)
                with open(f"outputs/mermaid_{i}.md", "w", encoding="utf-8") as f:
                    f.write(mermaid_data)
                print(f"📝 Saved mermaid_{i}.md")

                for e in entities:
                    all_data.append({"link": url, "tag": e.entity, "tag_type": e.attr_type})

        except Exception as e:
            print(f"⚠️ DSPy error on {url}: {e}")

    # Save CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv("outputs/tags.csv", index=False, encoding="utf-8")
        print("\n✅ Done! Saved outputs/tags.csv successfully.")
        print(df.head())
    else:
        print("\n⚠️ No data extracted. Check API key and connection.")

# --- 🔟 Run Script ---
if __name__ == "__main__":
    asyncio.run(main())
