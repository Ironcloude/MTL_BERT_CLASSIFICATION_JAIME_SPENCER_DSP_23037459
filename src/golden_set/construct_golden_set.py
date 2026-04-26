"""
Scrape articles for the golden evaluation set and extract archival baselines.
Saves results to data/golden_set/golden_articles.csv.

Sources: 
1. Contemporary: RSS feed (primary) => wayback machine.
2. Archival: Randomly sampled from raw ABP, Kaggle, and Webis datasets.

Re-running is safe — already-scraped URLs and archival extractions are skipped.

Largely informed and directed by Claude.
"""
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import os
import sys
import time
import feedparser
import requests
import trafilatura
import tldextract
import pandas as pd
import random
import csv
from bs4 import BeautifulSoup
import sys
import os
from data.utils import DATASET_CONFIGS, load_alignment_dataset

# Define browser user agents
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_PATH  = os.path.join(PROJECT_ROOT, "data/golden_set/golden_articles.csv")
# ABP is already encoded

ARTICLES_PER_OUTLET = 30
SCRAPE_DELAY        = 1.5    # seconds between requests
MIN_ARTICLE_WORDS   = 100
CDX_FROM            = "20250101"  # current articles only

# (display_name, domain, bias, rss_urls, allowed_path_fragments)
OUTLETS = [
    ("The Guardian", "theguardian.com", "left", [
        "https://www.theguardian.com/politics/rss",
        "https://www.theguardian.com/us-news/rss",
        "https://www.theguardian.com/commentisfree/rss",
        "https://www.theguardian.com/world/rss",
    ], ["/politics/", "/us-news/", "/uk-news/", "/commentisfree/", "/world/"]),

    ("BBC News", "bbc.com", "center", [
        "http://feeds.bbci.co.uk/news/politics/rss.xml",
        "http://feeds.bbci.co.uk/news/uk/rss.xml",
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
    ], ["/news/politics/", "/news/uk/", "/news/world/", "/news/articles/"]),

    # Newsweek: RSS is mostly lifestyle/sports junk — use section pages only
    ("Newsweek", "newsweek.com", "center", [], []),

    ("Daily Mail", "dailymail.co.uk", "right", [
        "https://www.dailymail.co.uk/news/index.rss",
    ], []),

    ("HuffPost", "huffpost.com", "left", [
        "https://www.huffpost.com/section/politics/feed",
    ], ["/entry/"]),

    ("New York Post", "nypost.com", "right", [
        "https://nypost.com/politics/feed/",
        "https://nypost.com/news/feed/",
    ], ["/2026/", "/2025/"]),
    
]



def registered_domain(url: str) -> str:
    """Return top-level domain for a URL (e.g. amp.ft.com → ft.com)."""
    try:
        top_level_domain = tldextract.extract(url)
        if top_level_domain.domain and top_level_domain.suffix:
            return f"{top_level_domain.domain}.{top_level_domain.suffix}".lower()
    except Exception:
        pass
    return ""


def on_domain(url: str, domain: str) -> bool:
    return registered_domain(url) == domain

def get_rss_urls(rss_feeds: list[str], domain: str, max_per_feed: int = 80) -> list[str]:
    """Fetch and domain-filter article URLs from one or more RSS feeds."""
    seen: set[str] = set()
    urls: list[str] = []
    for feed_url in rss_feeds:
        try:
            request = requests.get(feed_url, headers={
                "User-Agent": BROWSER_UA,
                "Accept": "application/rss+xml, application/xml, text/xml, */*;q=0.8",
            }, timeout=10)
            request.raise_for_status()
            entries = feedparser.parse(request.text).entries[:max_per_feed]
            for entry in entries:
                link = entry.get("link", "")
                if link and on_domain(link, domain) and link not in seen:
                    seen.add(link)
                    urls.append(link)
        except Exception as entry:
            print(f"  RSS error ({feed_url}): {entry}")
    print(f"  RSS: {len(urls)} on-domain URLs across {len(rss_feeds)} feed(s)")
    return urls


def looks_like_article(url: str) -> bool:
    """Heuristic: reject homepages, pagination, query-param junk, and media files."""
    if "?" in url or "#" in url:
        return False
    path = url.split("://", 1)[-1].split("/", 1)[-1] if "/" in url else ""
    if not path or path in ("", "/"):
        return False
    skip_exts = (".jpg", ".jpeg", ".png", ".gif", ".pdf", ".mp4", ".xml", ".rss")
    if any(path.lower().endswith(e) for e in skip_exts):
        return False
    return True


def get_section_page_urls(domain: str, section_urls: list[str]) -> list[str]:
    """Scrape article links from section/category index pages."""
    seen: set[str] = set()
    urls: list[str] = []
    for page_url in section_urls:
        try:
            r = requests.get(page_url, headers={"User-Agent": BROWSER_UA}, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/") and not href.startswith("//"):
                    href = f"https://www.{domain}" + href
                # Newsweek article URLs end with a numeric ID
                parts = href.rstrip("/").split("-")
                if domain in href and len(parts) > 2 and parts[-1].isdigit() and href not in seen:
                    seen.add(href)
                    urls.append(href)
            time.sleep(1)
        except Exception as exc:
            print(f"  Section page error ({page_url}): {exc}")
    print(f"  Section pages: {len(urls)} article URLs")
    return urls


def scrape_article(url: str, wayback_timestamp: str = "20260301") -> str | None:
    """Fetch a URL and extract clean article text via trafilatura.

    Falls back to the Wayback Machine's most recent snapshot when the
    live page blocks us (403/451) or returns no extractable text.

    Args:
        url: Article URL to scrape
        wayback_timestamp: CDX timestamp (YYYYMMDD) to search for snapshots. Default: 20260301.
                          For historic fallback, use earlier dates (e.g. "20200101").
    """
    # 1. Try the live page
    try:
        r = requests.get(url, headers={"User-Agent": BROWSER_UA}, timeout=15)
        r.raise_for_status()
        text = trafilatura.extract(
            r.text,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if text:
            return text
    except Exception:
        pass

    # 2. Wayback Machine fallback
    try:
        wb = requests.get(
            "http://archive.org/wayback/available",
            params={"url": url, "timestamp": wayback_timestamp},
            timeout=10,
        ).json()
        snap_url = wb.get("archived_snapshots", {}).get("closest", {}).get("url")
        if not snap_url:
            return None
        r2 = requests.get(snap_url, headers={"User-Agent": BROWSER_UA}, timeout=20)
        r2.raise_for_status()
        return trafilatura.extract(
            r2.text,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
    except Exception:
        return None

def get_archival_articles():
    """Extracts exactly 30 random articles per target outlet from the raw datasets."""
    print("\nLoading archival datasets to extract baselines...")    
    combined = load_alignment_dataset(DATASET_CONFIGS)
    
    # Clean text to ensure accurate word counts
    combined["text"] = combined["text"].astype(str).str.replace(r"\n+", " ", regex=True).str.strip()
    combined = combined[combined["text"].str.split().str.len() > MIN_ARTICLE_WORDS]
    combined = combined.drop_duplicates(subset="text")

    archival_rows = []
    
    for name, domain, bias, _, _ in OUTLETS:
        outlet_df = combined[combined["source"] == name]
        
        if len(outlet_df) == 0:
            print(f"  [WARNING] No archival articles found for {name}!")
            continue
            
        # Sample exactly 30, with a fixed random state for reproducibility
        sample_size = min(ARTICLES_PER_OUTLET, len(outlet_df))
        if sample_size < ARTICLES_PER_OUTLET:
            print(f"  [WARNING] {name} only has {sample_size} archival articles available.")
            
        sampled = outlet_df.sample(n=sample_size, random_state=42)
        
        for idx, row in sampled.iterrows():
            archival_rows.append({
                "outlet": name,
                "domain": domain,
                "bias": bias,
                "url": f"archive://{domain}/{idx}", # Dummy URL to satisfy formatting
                "text": row["text"],
                "timeframe": "archival"
            })
            
        print(f"  Extracted {sample_size} archival articles for {name}")
        
    return archival_rows

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load existing results so re-runs skip already-scraped URLs
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        seen_urls = set(existing["url"])
        rows = existing.to_dict("records")
        print(f"Loaded {len(existing)} existing articles.\n")
    else:
        seen_urls = set()
        rows = []

    for name, domain, bias, rss_feeds, allowed_path_fragments in OUTLETS:
        already = sum(1 for r in rows if r["domain"] == domain and r.get("timeframe") == "contemporary")
        needed  = ARTICLES_PER_OUTLET - already

        if needed <= 0:
            print(f"[SKIP] {name} — already have {already} contemporary articles")
            continue

        print(f"\n── {name} ({bias}) — need {needed} more ──")

        # 1. RSS (all configured feeds)
        candidates: list[str] = []
        if rss_feeds:
            candidates = get_rss_urls(rss_feeds, domain)

        # 1b. Apply path filter
        if allowed_path_fragments:
            before = len(candidates)
            candidates = [
                u for u in candidates
                if any(frag in u for frag in allowed_path_fragments)
            ]
            print(f"  Path filter: {before} → {len(candidates)} candidates")

        # 2. Section-page supplement (for outlets with weak/no RSS)
        if len(candidates) < needed and domain == "newsweek.com":
            print(f"  Supplementing with section pages...")
            section_urls = [
                "https://www.newsweek.com/politics",
                "https://www.newsweek.com/world",
                "https://www.newsweek.com/topic/us-politics",
                 "https://www.newsweek.com/topic/congress",
            ]
            section_new = [
                url for url in get_section_page_urls(domain, section_urls)
                if url not in set(candidates)
            ]
            candidates += section_new
            print(f"  Total candidates: {len(candidates)}")

        # 3. Scrape
        scraped = 0
        for url in candidates:
            if scraped >= needed:
                break
            if url in seen_urls:
                continue

            seen_urls.add(url)
            text = scrape_article(url)
            time.sleep(random.uniform(1.5, 4.0))

            if not text or len(text.split()) < MIN_ARTICLE_WORDS:
                print(f"  [skip] too short: {url[:70]}")
                continue

            rows.append({
                "outlet": name,
                "domain": domain,
                "bias":   bias,
                "url":    url,
                "text":   text,
                "timeframe": "contemporary"
            })
            scraped += 1
            print(f"  [{scraped}/{needed}] {url[:70]}")

        print(f"  Done — {scraped} new articles for {name}")

        # Save after each outlet
        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)

    # 2. Extract Archival Articles
    print("\n── PHASE 2: Extract Archival Articles ──")
    archival_needed = False
    for name, _, _, _, _ in OUTLETS:
        archival_count = sum(1 for r in rows if r["outlet"] == name and r.get("timeframe") == "archival")
        if archival_count < ARTICLES_PER_OUTLET:
            archival_needed = True
            break
            
    if archival_needed:
        archival_articles = get_archival_articles()

        # Only add the ones we don't already have (based on dummy URLs)
        for article in archival_articles:
            if article["url"] not in seen_urls:
                rows.append(article)
                seen_urls.add(article["url"])

        # 2b. Fallback: if any outlet still short, scrape historic Wayback snapshots
        print("\n── PHASE 2b: Fallback historic scrape for underfilled outlets ──")
        for name, domain, bias, rss_feeds, allowed_path_fragments in OUTLETS:
            archival_count = sum(1 for r in rows if r["outlet"] == name and r.get("timeframe") == "archival")
            if archival_count < ARTICLES_PER_OUTLET:
                shortfall = ARTICLES_PER_OUTLET - archival_count
                print(f"\n{name} short by {shortfall} archival articles — scraping historic Wayback snapshots...")

                # Re-fetch contemporary candidates
                candidates = []
                if rss_feeds:
                    candidates = get_rss_urls(rss_feeds, domain)
                if allowed_path_fragments:
                    candidates = [u for u in candidates if any(frag in u for frag in allowed_path_fragments)]

                # Scrape shortfall amount via Wayback Machine (2020 snapshots for historic content)
                scraped = 0
                for url in candidates:
                    if scraped >= shortfall:
                        break
                    if url in seen_urls:
                        continue

                    seen_urls.add(url)
                    # Try Wayback snapshot from 2020 (historic content)
                    text = scrape_article(url, wayback_timestamp="20200101")
                    time.sleep(random.uniform(1.5, 4.0))

                    if not text or len(text.split()) < MIN_ARTICLE_WORDS:
                        continue

                    rows.append({
                        "outlet": name,
                        "domain": domain,
                        "bias": bias,
                        "url": url,
                        "text": text,
                        "timeframe": "archival"  # Mark as archival (from Wayback 2020)
                    })
                    scraped += 1
                    print(f"  [{scraped}/{shortfall}] {url[:70]} (Wayback 2020)")

                print(f"  Supplemented {scraped}/{shortfall} articles for {name}")

        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    else:
        print("[SKIP] Archival articles already fully extracted.")

    # Final Verification
    df = pd.DataFrame(rows)
    print(f"\n{'='*50}")
    print(f"GOLDEN SET COMPLETE: {len(df)} total articles")
    print(f"{'='*50}")
    print(df.groupby(["outlet", "timeframe"])["url"].count().to_string())