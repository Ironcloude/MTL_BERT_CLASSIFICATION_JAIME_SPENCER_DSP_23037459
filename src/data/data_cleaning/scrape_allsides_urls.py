"""
Scrape AllSides outlet pages to find each outlet's website domain and current
bias rating, then save results to allsides_url.csv.
Uses Playwright + playwright_stealth to handle JS-rendered/Cloudflare-protected content.
Caches results so re-runs only scrape missing rows.
"""

import os, json, asyncio
from urllib.parse import urlparse
import pandas as pd
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from bs4 import BeautifulSoup

project_root    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALLSIDES_PATH   = os.path.join(project_root, "data/raw/allsides_outlet_ratings/allsides.csv")
ALLSIDES_OUTPUT = os.path.join(project_root, "data/raw/allsides_outlet_ratings/allsides_updated_2026.csv")
CACHE_PATH      = os.path.join(project_root, "data/raw/allsides_outlet_ratings/domain_cache.json")
SCRAPE_DELAY    = 0.5   # seconds between requests per worker
N_WORKERS       = 3     # parallel browser pages


def extract_domain(url):
    try:
        netloc = urlparse(str(url)).netloc
        return netloc.replace("www.", "").lower().strip()
    except Exception:
        return ""


async def scrape_outlet(page, allsides_url):
    """Navigate to an AllSides outlet page and return (domain, bias_rating)."""
    domain = ""
    rating = ""
    try:
        await page.goto(allsides_url, wait_until="domcontentloaded", timeout=15000)

        # Wait for the outlet link element (React-hydrated, not in SSR HTML)
        try:
            await page.wait_for_selector('a[id^="News-Source-Goto--"]', timeout=10000)
        except Exception:
            pass

        soup = BeautifulSoup(await page.content(), "html.parser")

        # Primary get tag containing outlet domain
        element = soup.find("a", id=lambda x: x and x.startswith("News-Source-Goto--"))
        if element and element.get("href") and "allsides.com" not in element["href"]:
            domain = extract_domain(element["href"])

        # Fallback: a.black-link with external href
        if not domain:
            for link in soup.find_all("a", class_="black-link"):
                href = link.get("href", "")
                if href.startswith("http") and "allsides.com" not in href:
                    domain = extract_domain(href)
                    break

        if not domain:
            print(f"  [DEBUG] no domain found for {allsides_url}")

        # Bias rating (server-rendered, always present)
        rating_element = await page.query_selector(".bias-rating")
        if rating_element:
            text = (await rating_element.inner_text() or "").strip().lower()
            if text:
                rating = text

        if not rating:
            print(f"  [DEBUG] no rating found for {allsides_url}")

    except Exception as e:
        print(f"  Error scraping {allsides_url}: {e}")

    return domain, rating


async def worker(browser, queue, results, lock, total):
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    while True:
        try:
            i, url = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        domain, rating = await scrape_outlet(page, url)
        async with lock:
            if domain and rating:
                results[url] = {"domain": domain, "rating": rating}
            done = total - queue.qsize()
            print(f"  [{done}/{total}] {url} => domain={domain or '—'}, rating={rating or '—'}")
        await asyncio.sleep(SCRAPE_DELAY)
    await context.close()


async def main():
    df = pd.read_csv(ALLSIDES_PATH)

    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached entries.")

    to_scrape = [url for url in df["allsides_page"] if url not in cache]
    print(f"{len(to_scrape)} pages to scrape ({len(cache)} already cached) with {N_WORKERS} workers...")

    queue = asyncio.Queue()
    for i, url in enumerate(to_scrape):
        await queue.put((i + 1, url))

    results = {}
    lock = asyncio.Lock()

    async with Stealth().use_async(async_playwright()) as pw:
        browser = await pw.chromium.launch(headless=True)
        workers = [worker(browser, queue, results, lock, len(to_scrape)) for _ in range(N_WORKERS)]
        await asyncio.gather(*workers)
        await browser.close()

    cache.update(results)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    df["domain"]       = df["allsides_page"].map(lambda u: cache.get(u, {}).get("domain", ""))
    df["scraped_bias"] = df["allsides_page"].map(lambda u: cache.get(u, {}).get("rating", ""))
    df["bias"] = df.apply(
        lambda row: row["scraped_bias"] if row["scraped_bias"] else row["bias"], axis=1
    )
    df = df.drop(columns=["scraped_bias"])

    df.to_csv(ALLSIDES_OUTPUT, index=False)
    print(f"\nSaved {len(df)} rows → {ALLSIDES_OUTPUT}")
    print(f"  Found domains: {(df['domain'] != '').sum()} / {len(df)}")


if __name__ == "__main__":
    asyncio.run(main())
