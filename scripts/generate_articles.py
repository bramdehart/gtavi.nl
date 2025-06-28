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
            logger.info("✅ Cookie banner accepted")
        except PlaywrightTimeout:
            logger.info("ℹ️ No cookie button found")

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


def fix_image_paths(markdown: str) -> str:
    return markdown.replace("(assets/images/1200", "(/assets/images/1200")


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
    lines = markdown.splitlines()
    for line in lines:
        match = re.match(r'^#+\s+(.*)', line)
        if match:
            header = match.group(1).strip()
            break
    else:
        raise ValueError("No header found in markdown")

    slug = header.lower()
    slug = re.sub(r'\s+', '_', slug)
    slug = re.sub(r'[^\w\-]', '', slug)
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

    images_dir = Path("assets/images/1200")
    image_files = [f for f in images_dir.glob("*.jpg") if f.is_file()]
    image_url = f"/assets/images/1200/{random.choice(image_files).name}" if image_files else None

    input_text = extract_text_from_url(source_url)

    system_prompt = (
        "Je bent een redacteur die nieuws herschrijft op een unieke, toegankelijke en feitelijke manier. "
        "Gebruik geen letterlijke zinnen. Vat het origineel samen en herschrijf het vloeiend. Voeg eventueel context toe. "
        "De doelgroep is geïnteresseerd in GTA VI en wil heldere artikelen zonder clickbait."
        "Je hebt een statische Jekyll website en dient rekening te houden met de limieten die dit met zich meebrengt."
        "Het geschreven artikel stuur je daarom terug in markdown format"
    )

    user_prompt = (
        "Schrijf een herschreven, uniek artikel in het Nederlands van dit nieuws, met een pakkende intro en goede structuur."
        "Schrijf het artikel in markdown formaat en houd rekening met de eisen die Jekyll met zich meebrengt."
        "Zorgt ervoor dat het artikel een header kop bevat met de onderdelen title, date, categories en tags."
        f"Als datum kies je het volgende waarde, die je schrijft in het formaat yyyy-mm-dd hh:mm:ss -200: {publication_date}."
        "Kies één of meerdere categorieën uit de volgende opsomming: Nieuws, Geruchten, Leaks, Gameplay, Verhaal, Locatie, Personages, Voertuigen, Multiplayer, Techniek, Release, PlayStation, Xbox, PC, \"Rockstar Games\"."
        "Kies één of meerdere tags uit de volgende opsomming: \"vice city\", lucia, jason, trailer, map, wapens, npc, online, \"rage engine\", \"rockstar games\", ps5, \"xbox series x\", \"xbox series s\", pc."
        "Vervolgens vul je de markdown met je eigen geschreven artikel."
        "Begin het artikel altijd met een header (#)."
        "Plaats nooit externe afbeeldingen in je markdown."
        "Plaats nooit meer dan 1 afbeelding in je markdown."
        f"Kies deze afbeelding URL: {image_url}."
        "Zorg ervoor dat de afbeelding URL altijd begint met een enkele leading slash (/)."
        f"Hier volgt het artikel dat moet worden herschreven:\n{input_text}"
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
                if entry.source.href in ["https://www.gamekings.tv", "https://metro.co.uk"]:
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
