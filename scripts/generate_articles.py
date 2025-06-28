import feedparser
import json
import logging
import os
import random
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI
from playwright.sync_api import sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

def extract_text_from_url(url: str) -> str:
     
      with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=30000)

        # Try to click the cookie wall button
        try:
            page.wait_for_selector("button:has-text(\"Accept\")", timeout=5000)
            page.click("button:has-text(\"Accept\")")
            logger.info("✅ Google cookie banner accepted")
        except PlaywrightTimeout:
            logger.info("ℹ️ No Google cookie button found")

        # Try to wait for main content
        try:
            page.wait_for_selector("article, .article, .content", timeout=10000)
            content_html = page.inner_html("article, .article, .content")
            text = BeautifulSoup(content_html, "html.parser").get_text(separator="\n", strip=True)
            if text:
                return text
        except PlaywrightTimeout:
            logger.info("ℹ️ No <article> or .article/.content found — using fallback")

        # Fallback: use the entire body and BeautifulSoup to extract all text
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        # Search within text elements
        text_elements = soup.find_all(['p', 'div', 'span', 'section'])
        text = '\n'.join(
            el.get_text(strip=True)
            for el in text_elements
            if el.get_text(strip=True)
        )

        # Latest fallback: meta description
        if not text:
            meta = soup.find('meta', attrs={'name': 'description'})
            if meta and 'content' in meta.attrs:
                text = meta['content']

        browser.close()
        return text.strip() 

def extract_date_only(markdown: str) -> str:
    match = re.search(r'^date:\s+(\d{4}-\d{2}-\d{2})', markdown, re.MULTILINE)
    if match:
        return match.group(1)
    return None

def fix_image_paths(markdown: str) -> str:
    return markdown.replace("(assets/img/1200", "(/assets/img/1200")


def strip_markdown_code_fence(md: str) -> str:
    md = md.strip()
    if md.startswith("```markdown"):
        md = md[len("```markdown"):].lstrip()
    elif md.startswith("```"):
        md = md[len("```"):].lstrip()
    if md.endswith("```"):
        md = md[:-3].rstrip()
    return md


def slug_from_markdown(markdown: str) -> str:
    lines = markdown.strip().splitlines()
    for line in lines:
        match = re.match(r'^title:\s+(.*)', line, re.IGNORECASE)
        if match:
            header = match.group(1).strip()
            break
    else:
        raise ValueError("No header found in markdown")

    slug = header.lower()
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'[^\w\-]', '', slug)
    extracted_date = extract_date_only(markdown)
    if (extracted_date):
        return extracted_date + '-' + slug
    return slug


def save_article_and_embedding(data: dict):
    title_slug = data["title_slug"]
    article_markdown = data["article_markdown"]
    embedding = data["embedding"]

    Path("_posts").mkdir(parents=True, exist_ok=True)
    Path("embeddings").mkdir(parents=True, exist_ok=True)

    with open(f"_posts/{title_slug}.md", "w", encoding="utf-8") as f:
        f.write(article_markdown)

    with open(f"embeddings/{title_slug}.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(embedding))

    logger.info(f"✅ Saved: _posts/{title_slug}.md and embeddings/{title_slug}.txt")


def calculate_similarity(vec1, vec2) -> float:
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def is_similar_to_existing(article_embedding: list[float], threshold: float = 0.9) -> bool:
    embeddings_directory = "embeddings"
    if not os.path.exists(embeddings_directory):
        return False
    for filename in os.listdir(embeddings_directory):
        try:
            with open(os.path.join(embeddings_directory, filename), 'r') as f:
                embedding = json.load(f)
                if calculate_similarity(article_embedding, embedding) >= threshold:
                    return True
        except Exception:
            continue
    return False


def is_similar_to_generated_articles(new_article: dict, generated_articles: list[dict], threshold: float = 0.9) -> bool:
    new_embedding = new_article['embedding']
    for article in generated_articles:
        if calculate_similarity(new_embedding, article['embedding']) >= threshold:
            return True
    return False


def generate_article(source_url: str, publication_date: str, generated_articles: list[dict] = []) -> dict:
    api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    embeddings_client = OpenAI(api_key=api_key)

    images_dir = Path("assets/img/1200")
    image_files = [f for f in images_dir.glob("*.jpg") if f.is_file()]
    image_url = f"/assets/img/1200/{random.choice(image_files).name}" if image_files else None

    input_text = extract_text_from_url(source_url)

    system_prompt = (
        "Je bent een redacteur die nieuws herschrijft op een unieke, toegankelijke en feitelijke manier. "
        "Gebruik geen letterlijke zinnen uit de bron. Vat het origineel samen en herschrijf het vloeiend, met eventueel extra context indien relevant. "
        "De doelgroep is geïnteresseerd in GTA VI en waardeert heldere, goed gestructureerde artikelen zonder clickbait. "
        "Het artikel wordt gepubliceerd op een statische Jekyll-website, dus houd rekening met de beperkingen die dat met zich meebrengt. "
        "Stuur het herschreven artikel terug in markdown-formaat."
    )

    user_prompt = (
        "Schrijf een uniek, herschreven artikel in het Nederlands op basis van onderstaand nieuws. "
        "Zorg voor een pakkende intro, een duidelijke structuur en een toegankelijke schrijfstijl.\n\n"
        "Het artikel moet worden opgemaakt in markdown-formaat, geschikt voor gebruik in een Jekyll-blog.\n\n"
        "Zorgt ervoor dat de markdown een header kop bevat met de onderdelen title, date, categories tags en image.path zoals verwacht in een Jekyll post."
        "Plaats de titel altijd binnen quotes (\") om corrupte markdown te voorkomen"
        f"De waarde van image.path is {image_url}. Zorg dat het pad altijd begint met een `/`"
        f"Gebruik voor date het formaat: yyyy-mm-dd hh:mm:ss +200, met als waarde: {publication_date})\n"
        "Kies één of meerdere categorieën uit deze lijst:\n"
        "Nieuws, Geruchten, Leaks, Gameplay, Verhaal, Locatie, Personages, Voertuigen, Multiplayer, Techniek, "
        "Release, PlayStation, Xbox, PC, \"Rockstar Games\".\n\n"
        "Kies één of meerdere tags uit deze lijst:\n"
        "\"vice city\", lucia, jason, trailer, map, wapens, npc, online, \"rage engine\", \"rockstar games\", ps5, "
        "\"xbox series x\", \"xbox series s\", pc.\n\n"
        "Let op:\n"
        "- Noteer categorieën en tags als geldige YAML-arrays met vierkante haken ([])\n"
        "- Gebruik aanhalingstekens (\"\") rond elk item met een spatie, bijvoorbeeld \"gta 6\"\n\n"
        "De header ziet er bijvoorbeeld zo uit:\n"
        "---"
        "title: Vice City Voltage – De Elektrische Ford Capri en de Verwachtingen voor GTA 6"
        "date: 2025-06-28 18:49:50 +200"
        "categories: [\"Nieuws\", \"Rockstar Games\", \"Locatie\"]"
        "tags: [\"vice city\", \"rockstar games\", \"ps5\", \"xbox series x\"]"
        "image:"
        "   path: /assets/img/1200/Vice_City_01.jpg"
        "---"
        "Vervolgens vul je de markdown met je eigen geschreven artikel."
        "Gebruik geen afbeeldingen in het artikel, enkel in de header.\n"
        "Gebruik de onderstaande tekst als bron voor het herschreven artikel:\n"
        f"{input_text}"
    )


    article_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=3000
    )

    article_markdown = strip_markdown_code_fence(article_response.choices[0].message.content)
    article_markdown = fix_image_paths(article_markdown)
    title_slug = slug_from_markdown(article_markdown)

    embedding_response = embeddings_client.embeddings.create(
        model="text-embedding-3-small",
        input=article_markdown
    )

    embedding = embedding_response.data[0].embedding

    return {
        "title_slug": title_slug,
        "article_markdown": article_markdown,
        "embedding": embedding
    }


def fetch_latest_gta_news():
    rss_url = "https://news.google.com/rss/search?q=GTA+6&hl=en&gl=US&ceid=US:en"
    feed = feedparser.parse(requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}).text)

    news_items = []
    for entry in sorted(feed.entries, key=lambda e: datetime(*e.published_parsed[:6]), reverse=True):
        if hasattr(entry, "link") and hasattr(entry, "published"):
            if hasattr(entry, "source") and hasattr(entry.source, "href"):
                if entry.source.href in ["https://www.gamekings.tv", "https://metro.co.uk", "https://timesofindia.indiatimes.com"]:
                    continue
            news_items.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published
            })
    return news_items


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    article_count = int(os.getenv('ARTICLE_COUNT', '1'))
    news_items = fetch_latest_gta_news()

    generated_articles = []
    articles_processed = 0

    for news_item in news_items:
        if articles_processed >= article_count:
            break
        logger.info(f"Processing article: {news_item['title']}")
        article = generate_article(news_item["link"], news_item["published"])
        if article and not is_similar_to_existing(article['embedding']) and not is_similar_to_generated_articles(article, generated_articles):
            save_article_and_embedding(article)
            generated_articles.append(article)
            articles_processed += 1
            logger.info(f"✅ Saved unique article {articles_processed}/{article_count}")
        else:
            logger.info(f"⏩ Skipped article due to similarity: {news_item['title']}")

    # print(json.dumps(generated_articles, indent=2))
