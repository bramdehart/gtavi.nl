import feedparser
import json
import logging
import os
import random
import re
import requests
import time
from bs4 import BeautifulSoup
from datetime import datetime
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def accept_cookie_if_present(driver, timeout=5):
    xpath = (
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept') or "
        "contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'akkoord') or "
        "contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]"
    )

    def try_accept_in_context(context_desc="main content"):
        try:
            button = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            driver.execute_script("arguments[0].click();", button)
            print(f"✅ Cookie accepted in {context_desc}")
            return True
        except TimeoutException:
            print(f"ℹ️ No cookie button found in {context_desc}")
            return False

    # Check iframes
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    print(f"🔍 Found {len(iframes)} iframe(s)")

    for index, frame in enumerate(iframes):
        try:
            driver.switch_to.frame(frame)
            if try_accept_in_context(f"iframe {index}"):
                driver.switch_to.default_content()
                return True
        except Exception as e:
            print(f"⚠️ Error accessing iframe {index}: {e}")
        finally:
            driver.switch_to.default_content()

    # If not in any iframe, try main page
    return try_accept_in_context("main page")

def fix_image_paths(markdown: str) -> str:
    return markdown.replace("(assets/images/1200", "(/assets/images/1200")

def strip_markdown_code_fence(md: str) -> str:
    md = md.strip()

    # Remove starting ```markdown or ```
    if md.startswith("```markdown"):
        md = md[len("```markdown"):].lstrip()
    elif md.startswith("```"):
        md = md[len("```"):].lstrip()

    # Remove ending ```
    if md.endswith("```"):
        md = md[:-3].rstrip()

    return md

def save_article_and_embedding(data: dict):
    title_slug = data["title_slug"]
    article_markdown = data["article_markdown"]
    embedding = data["embedding"]

    # Ensure directories exist
    Path("_posts").mkdir(parents=True, exist_ok=True)
    Path("embeddings").mkdir(parents=True, exist_ok=True)

    # Save article markdown
    post_path = Path(f"_posts/{title_slug}.md")
    with open(post_path, "w", encoding="utf-8") as f:
        f.write(article_markdown)

    # Save embedding as plain text
    embedding_path = Path(f"embeddings/{title_slug}.txt")
    with open(embedding_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(embedding))

    print(f"✅ Saved: {post_path} and {embedding_path}")

def calculate_similarity(vec1, vec2) -> float:
    """Calculate cosine similarity between two vectors."""
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def is_similar_to_existing(article_embedding: list[float], threshold: float = 0.85) -> bool:
    # Check if the new article is similar to any existing article
    embeddings_directory = "embeddings"
    if not os.path.exists(embeddings_directory):
        # No existing articles
        return False
    
    # Load embeddings from all existing articles
    existing_embeddings = []
    for filename in os.listdir(embeddings_directory):
        try:
            with open(os.path.join(embeddings_directory, filename), 'r') as f:
                embedding = json.load(f)
                existing_embeddings.append(embedding)
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")

    
    # Compare with all existing embeddings
    for existing_embedding in existing_embeddings:
        similarity = calculate_similarity(article_embedding, existing_embedding)
        if similarity >= threshold:
            logger.info(f"Found similar article with similarity: {similarity}")
            return True
    
    return False

def is_similar_to_generated_articles(new_article: dict, generated_articles: list[dict], threshold: float = 0.85) -> bool:
    # Check if the new article is similar to any already generated articles.
    if not generated_articles:
        return False
    
    new_embedding = new_article['embedding']
    for article in generated_articles:
        similarity = calculate_similarity(new_embedding, article['embedding'])
        if similarity >= threshold:
            logger.info(f"Found similar article in generated articles with similarity: {similarity}")
            return True
    
    return False

def slug_from_markdown(markdown: str) -> str:
    """
    Extract the first markdown header and convert it to a URL-friendly slug.
    
    Rules:
    - All lowercase
    - Spaces → underscores
    - Remove special characters (except underscores and hyphens)
    - Raise ValueError if no header is found
    """
    lines = markdown.splitlines()
    
    # Find first markdown header (e.g. "# My Title")
    for line in lines:
        match = re.match(r'^#+\s+(.*)', line)
        if match:
            header = match.group(1).strip()
            break
    else:
        raise ValueError("No header found in markdown")

    # Slugify: lowercase, replace spaces, remove unwanted characters
    slug = header.lower()
    # spaces → underscores
    slug = re.sub(r'\s+', '_', slug)
    # remove non-word chars (keep _ and -)
    slug = re.sub(r'[^\w\-]', '', slug)
    return slug

def generate_article(source_url: str, publication_date: str, generated_articles: list[dict] = []) -> dict:
    # Define OpenAI API client
    api_key = os.environ["OPENAI_API_KEY"]
    if not api_key:
        logger.error("OpenAI API key is empty or not set")
        raise ValueError("OpenAI API key is empty or not set")
    
    client = OpenAI(api_key=api_key)
    embeddings_client = OpenAI(api_key=api_key)

    try:
        # Select a random image
        images_dir = Path("assets/images/1200")
        image_files = [f for f in images_dir.glob("*.jpg") if f.is_file()]

        if image_files:
            random_image = random.choice(image_files)
            image_url = f"/assets/images/1200/{random_image.name}"
        else:
            image_url = None
            logger.error("No images found in assets/images/1200 directory")
            raise FileNotFoundError("No images found in assets/images/1200 directory")

        # Define prompt
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

        # Execute prompt
        logger.info("Making OpenAI API request for URL:", source_url)
        article_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=3000
        )

        logger.info("Successfully received OpenAI response:", article_response)
        article_markdown = strip_markdown_code_fence(article_response.choices[0].message.content)
        article_markdown = fix_image_paths(article_markdown)
        title_slug = slug_from_markdown(article_markdown)

        # Generate embedding for the article
        logger.info("Generating embedding for article")
        embedding_response = embeddings_client.embeddings.create(
            model="text-embedding-3-small",
            input=article_markdown
        )

        embedding = embedding_response.data[0].embedding
        logger.info("Successfully generated embedding:", embedding)

        return {
            "title_slug": title_slug,
            "article_markdown": article_markdown,
            "embedding": embedding
        }
    except Exception as e:
        logger.error(f"Error generating article: {str(e)}")
        raise

def fetch_latest_gta_news():
    # Fetch multiple GTA news items from Google News RSS feed.
    # Google News RSS-feed with search term "GTA 6" in Dutch

    rss_url = "https://news.google.com/rss/search?q=GTA+6&hl=en&gl=US&ceid=US:en"
    
    # Disable verify on local machine. Always enable on production.
    # feed = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, verify=False) 
    feed = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, verify=True)
    feed = feedparser.parse(feed.text)

    sorted_items = sorted(
        feed.entries,
        key=lambda entry: datetime(*entry.published_parsed[:6]),
        reverse=True
    )
    
    if not sorted_items:
        logger.error("No news items found in RSS feed")
        raise ValueError("RSS feed returned no news items")

    # Process all entries
    news_items = []
    for entry in sorted_items:
        item = {
            "title": entry.title if hasattr(entry, 'title') else None,
            "link": entry.link if hasattr(entry, 'link') else None,
            "published": entry.published if hasattr(entry, 'published') else None,
            "source_url": entry.source.href if hasattr(entry, 'source') and hasattr(entry.source, 'href') else None
        }

        # Skip articles from gamekings.tv
        if item["source_url"] == "https://www.gamekings.tv" or item["source_url"] == "https://metro.co.uk":
            continue

        # Only include items with valid links and publication date
        if item["link"] and item["published"]:
            news_items.append(item)

    logger.info(f"Found {len(news_items)} news items")
    return news_items

def extract_text_from_url(url: str) -> str:
    # Scrapes the main text of an article from a URL using Selenium.
    try:
        # Configure Chrome options
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless') # Run in headless mode
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36')
        
        # Start Chrome driver using GitHub Actions environment
        driver = webdriver.Chrome(
            options=chrome_options
        )

        try:
            # Navigate to the URL
            driver.get(url)

            try:
                # Click on the "Accept all" button if it's visible (Google cookie blocker)
                accept_btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept all') or contains(., 'Akkoord') or contains(., 'Accept')]"))
                )
                accept_btn.click()
                logger.info("✅ Google cookie wall accepted")
            except Exception:
                logger.info("ℹ️ No Google cookie wall or button not found")
            
            # Wait for the article page to load after Google cookie banner agreement
            logger.info("Wait for the article page to load after Google cookie banner agreement")
            time.sleep(10)
            # Agree to cookie banners on the article page
            logger.info("Agree to cookie banners on the article page")
            accept_cookie_if_present(driver, 5)
            # Wait for the article page to load after article cookie banner agreement
            logger.info('Wait for the article page to load after article cookie banner agreement')
            time.sleep(10)
            logger.info('Done waiting 10 seconds')

            try:
                # Wait for article content or main content to be visible
                article = wait = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'article, .article, .content'))
                )
            except TimeoutException:
                # If article not found, try to find all visible text
                article = None
            
            # Get the page source and parse with BeautifulSoup
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract text
            if article:
                text = article.text.strip()
            else:
                # Try to find content by looking at all visible text elements
                text_elements = soup.find_all(['p', 'div', 'span', 'section'])
                text = '\n'.join(element.get_text(strip=True) for element in text_elements 
                               if element.get_text(strip=True))
                
                # If still no text, try to find content in meta tags
                if not text:
                    meta_description = soup.find('meta', attrs={'name': 'description'})
                    if meta_description and 'content' in meta_description.attrs:
                        text = meta_description['content']
            
            # Clean up the text
            text = text.strip()
            
            # Log debug information
            logger.debug(f"\nURL: {url}")
            logger.debug(f"Scraped text length: {len(text)} characters")
            logger.debug(f"First 200 characters: {text[:200]}")
            
            return text
            
        finally:
            # Always close the driver
            driver.quit()
            
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Get the number of articles to generate from environment variable
    article_count = int(os.getenv('ARTICLE_COUNT', '1'))
    
    # Get all GTA news items
    news_items = fetch_latest_gta_news()
    if not news_items:
        logger.error("No news items found")
        exit(1)

    generated_articles = []
    articles_processed = 0
    
    # Process news items until we have enough unique articles or run out of items
    for news_item in news_items:
        if articles_processed >= article_count:
            break
        logger.info(f"Processing article: {news_item['title']}")
        article = generate_article(news_item["link"], news_item["published"])
        if article:
            # Check against both existing and already generated articles
            if is_similar_to_existing(article['embedding']) or \
               is_similar_to_generated_articles(article, generated_articles):
                logger.info(f"Skipping article: {news_item['title']} - Similar content found")
                continue

            # Save unique article and continue
            save_article_and_embedding(article)
            generated_articles.append(article)
            articles_processed += 1
            logger.info(f"Found unique article {articles_processed}/{article_count}")
        else:
            logger.info(f"Skipping article: {news_item['title']}")
    
    # Print all generated articles as a JSON array
    print(json.dumps(generated_articles))