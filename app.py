import streamlit as st
import requests
import json
import time
from datetime import datetime
from bs4 import BeautifulSoup
import newspaper # For newspaper.build and newspaper.Config
from newspaper import Article, ArticleException
import random
import os
import nltk # For NLTK resource download
import re
import google.generativeai as genai # For Google Gemini API
import threading
import queue
import io # For traceback string
import traceback as tb_module # Alias to avoid conflict

# --- Google OAuth and API Client ---
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- Global Configuration (Defaults, can be overridden by UI) ---
DEFAULT_MIN_POST_WORD_COUNT = 1500
DEFAULT_MAX_ARTICLES_TO_COMBINE = 3
POSTED_URLS_FILE = "posted_news_urls.txt"
TOKEN_FILE = 'token.json' # For storing OAuth tokens
SCOPES = ['https://www.googleapis.com/auth/blogger']

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

# --- Global variables for API keys (to be set by Streamlit UI) ---
# These are used by functions that might not have direct access to st.session_state
_YOUTUBE_API_KEY_GLOBAL = None
_GEMINI_API_KEY_GLOBAL = None # This is the one genai will be configured with
_BLOG_ID_GLOBAL = None

# --- Threading and Logging Setup ---
stop_event_global = threading.Event()
log_queue_global = queue.Queue()

def add_log(message):
    """Adds a message to the global log queue."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_queue_global.put(f"[{timestamp}] {message}")

# --- Helper Functions (Adapted from original, using add_log) ---
def get_random_user_agent():
    return random.choice(USER_AGENTS)

def count_words(text_content):
    if not text_content: return 0
    words = re.findall(r'\b\w+\b', text_content.lower())
    return len(words)

def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        add_log("NLTK 'punkt' resource already downloaded.")
    except LookupError:
        add_log("NLTK 'punkt' resource not found. Downloading...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.data.find('tokenizers/punkt') # Verify download
            add_log("NLTK 'punkt' downloaded and verified successfully.")
        except Exception as e:
            add_log(f"Error during NLTK 'punkt' download: {e}. Manual download may be required.")

def load_posted_urls():
    if not os.path.exists(POSTED_URLS_FILE): return set()
    try:
        with open(POSTED_URLS_FILE, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    except Exception as e:
        add_log(f"Error loading posted URLs file: {e}")
        return set()

def save_posted_url(url):
    try:
        with open(POSTED_URLS_FILE, 'a', encoding='utf-8') as f:
            f.write(url + '\n')
        add_log(f"Saved URL to posted list: {url}")
    except Exception as e:
        add_log(f"Error saving posted URL {url}: {e}")


# --- AI Content Enhancement (Adapted) ---
def configure_gemini_api_runtime(api_key_val):
    global _GEMINI_API_KEY_GLOBAL
    if api_key_val and api_key_val != "AIzaSyCrsJE4ZR_RxBvy2rlGjWalVkuUudKTm0c": # Check against placeholder
        try:
            genai.configure(api_key=api_key_val)
            _GEMINI_API_KEY_GLOBAL = api_key_val
            add_log("Google Gemini API configured successfully for runtime.")
            return True
        except Exception as e:
            add_log(f"Error configuring Gemini API at runtime: {e}. AI features will be disabled.")
            _GEMINI_API_KEY_GLOBAL = None
            return False
    else:
        add_log("Gemini API Key not set or is placeholder. AI features will be disabled for runtime.")
        _GEMINI_API_KEY_GLOBAL = None
        return False

def enhance_article_with_ai(article_data):
    if not _GEMINI_API_KEY_GLOBAL:
        add_log("Gemini API key not available globally. Skipping AI content enhancement.")
        return article_data

    add_log(f"Attempting AI enhancement for: {article_data['title'][:50]}...")
    # Using gemini-1.5-pro-latest as per original script.
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    original_title = article_data['title']
    original_text = article_data['text']
    
    prompt = f"""
You are an expert tech news editor and SEO specialist, writing for an audience passionate about AI, technology, and gadgets.
Given the following news article title and text:

Original Title: "{original_title}"
Original Text:
"{original_text}"

Please perform the following tasks and provide the output ONLY as a valid JSON object:
1.  Rewrite the title to be highly engaging, SEO-friendly, and exciting for tech enthusiasts (max 12 words, ideally 7-10 words). Ensure it accurately reflects the core content. Key: "seo_title".
2.  Rewrite the entire article text. Transform it into a captivating and highly readable piece for a tech-savvy audience.
    *   Use vivid language and an engaging narrative structure.
    *   Correct any grammatical errors, improve clarity, flow, and readability.
    *   Preserve ALL key factual information and main points from the original text.
    *   **Crucially, DO NOT mention the original news source, its website, or any specific branding from the source within the rewritten text. Focus solely on the information and present it neutrally.**
    *   Maintain a professional, informative, yet exciting tone suitable for a leading tech blog.
    *   Ensure the rewritten text is well-structured and avoids unnecessary repetition.
    Key: "enhanced_text".
3.  Generate a list of 5 to 7 highly relevant SEO keywords or tags (as a list of strings), specifically targeting AI, technology, and gadget-related searches. Key: "seo_keywords".
4.  Create a compelling meta description (around 150-160 characters) summarizing the article's key points in a way that attracts clicks from tech readers. Key: "meta_description".

JSON Output Structure Example:
{{
  "seo_title": "AI-Powered Gadget Revolution: Tech's Next Leap Forward!",
  "enhanced_text": "The completely rewritten, engaging, and SEO-friendly article text, focusing on tech innovations, will appear here. All facts are preserved, but the language is more vivid and tailored for tech enthusiasts, with no mention of the original publication...",
  "seo_keywords": ["AI innovation", "latest gadgets", "tech breakthroughs", "consumer electronics", "futuristic tech"],
  "meta_description": "Discover the latest AI-driven gadget revolution! Explore groundbreaking technology and futuristic innovations set to change our world. Click for the full story."
}}

Ensure your response is ONLY the JSON object.
"""
    try:
        response = model.generate_content(prompt)
        
        json_response_text = response.text.strip()
        if json_response_text.startswith("```json"):
            json_response_text = json_response_text[len("```json"):]
        if json_response_text.endswith("```"):
            json_response_text = json_response_text[:-len("```")]
        json_response_text = json_response_text.strip()

        ai_output = json.loads(json_response_text)

        enhanced_data = article_data.copy()
        enhanced_data['title'] = ai_output.get('seo_title', original_title).strip()
        enhanced_data['text'] = ai_output.get('enhanced_text', original_text).strip()
        enhanced_data['ai_keywords'] = ai_output.get('seo_keywords', [])
        enhanced_data['ai_meta_description'] = ai_output.get('meta_description', "").strip()
        
        enhanced_data['word_count'] = count_words(enhanced_data['text'])

        add_log(f"AI Enhancement successful. New title: {enhanced_data['title'][:50]}... New word count: {enhanced_data['word_count']}")
        return enhanced_data

    except json.JSONDecodeError as jde:
        add_log(f"AI Enhancement failed: Could not decode JSON from AI response. Error: {jde}")
        add_log(f"Problematic AI response part: '{response.text[:500]}...'")
        return article_data
    except Exception as e:
        add_log(f"AI Enhancement failed: {e}")
        return article_data

# --- Core Logic (Adapted) ---
def fetch_news_from_duckduckgo(query="popular news headlines", num_results=3):
    add_log(f"Searching DuckDuckGo for: \"{query}\" (expecting {num_results} results)")
    search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    headers = {'User-Agent': get_random_user_agent()}
    news_items = []
    try:
        response = requests.get(search_url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('a', class_='result__a')

        for result in results:
            title = result.get_text(strip=True)
            url = result.get('href')

            if url and url.startswith("http") and "duckduckgo.com" not in url:
                if url.startswith("https://duckduckgo.com/l/"): # Handle DDG redirect links
                    parsed_url_qs = requests.utils.urlparse(url).query
                    actual_url_match = requests.utils.parse_qs(parsed_url_qs).get('uddg')
                    if actual_url_match:
                        url = actual_url_match[0]

                if title and url.startswith('http'):
                    tech_keywords = ['ai', 'tech', 'gadget', 'robot', 'software', 'hardware', 'innovation', 'digital', 'crypto', 'cyber', 'platform', 'device', 'wearable', 'ar', 'vr', 'metaverse', 'nvidia', 'intel', 'amd', 'apple', 'google', 'microsoft', 'amazon', 'tesla', 'spacex']
                    is_relevant = any(keyword in title.lower() for keyword in tech_keywords) or \
                                  any(keyword in url.lower() for keyword in tech_keywords)
                    
                    if query.lower().count(" ") < 3 or is_relevant:
                        news_items.append({'title': title, 'url': url})
                        if len(news_items) >= num_results:
                            break
        
        if not news_items:
            add_log(f"No usable news results found on DuckDuckGo matching tech focus for query: \"{query}\"")
        else:
            add_log(f"Found {len(news_items)} potential sources/articles from DDG for \"{query}\".")
        return news_items

    except requests.exceptions.RequestException as e:
        add_log(f"Error fetching news from DuckDuckGo for \"{query}\": {e}")
        return []
    except Exception as e:
        add_log(f"Error parsing DuckDuckGo results for \"{query}\": {e}")
        return []

def scrape_article_content(article_url):
    add_log(f"Scraping article: {article_url}")
    try:
        config = newspaper.Config()
        config.request_timeout = 25
        config.browser_user_agent = get_random_user_agent()
        config.memoize_articles = False
        # config.verbose = True # newspaper3k can be noisy, keep this off for Streamlit logs

        article = Article(article_url, config=config)
        article.download()
        if stop_event_global.wait(random.uniform(0.5, 1.5)): return None # Interruptible sleep
        article.parse()

        article_word_count = count_words(article.text)

        if not article.title or not article.text or article_word_count < 100: # Min content length for a single article
            add_log(f"Failed to extract sufficient content from {article_url}. Title: '{article.title}', Word Count: {article_word_count}")
            return None

        keywords = []
        summary = ""
        try:
            article.nlp()
            keywords = list(article.keywords) if article.keywords else []
            summary = article.summary
            if (not summary or len(summary) < 50) and article.text:
                summary_sentences = article.text.split('.')[:3] # Take first 3 sentences as summary
                summary = '. '.join(s.strip() for s in summary_sentences if s.strip()).strip()
                if summary: summary += '.'
        except Exception as nlp_e:
            add_log(f"NLP processing failed for {article_url}: {nlp_e}. Generating basic summary.")
            if article.text: # Fallback summary
                summary_sentences = article.text.split('.')[:3]
                summary = '. '.join(s.strip() for s in summary_sentences if s.strip()).strip()
                if summary: summary += '.'
        
        all_images = list(article.images)

        add_log(f"Successfully scraped '{article.title}' ({article_word_count} words, {len(all_images)} images found) from {article_url}")
        return {
            'title': article.title, 'text': article.text, 'summary': summary,
            'authors': article.authors if article.authors else [],
            'publish_date': article.publish_date.strftime('%Y-%m-%d') if article.publish_date else "N/A",
            'top_image_url': article.top_image if article.top_image else None,
            'all_images': all_images, 'keywords': keywords, 'url': article_url,
            'word_count': article_word_count, 'ai_keywords': [], 'ai_meta_description': ""
        }
    except ArticleException as e:
        add_log(f"Newspaper3k ArticleException scraping {article_url}: {e}")
        return None
    except Exception as e:
        add_log(f"Generic error scraping {article_url}: {e}")
        return None

def get_sufficient_article_content(initial_article_data, posted_urls_set, min_post_word_count, max_articles_to_combine):
    all_articles_data = [initial_article_data]
    current_total_words = initial_article_data.get('word_count', 0)

    if current_total_words >= min_post_word_count:
        add_log(f"Initial article already meets word count: {current_total_words} words.")
        return all_articles_data

    add_log(f"Initial article word count ({current_total_words}) is less than target ({min_post_word_count}). Trying to find related articles...")

    search_terms = []
    effective_keywords = initial_article_data.get('ai_keywords', initial_article_data.get('keywords', []))
    if effective_keywords: search_terms.extend([kw for kw in effective_keywords[:2] if kw.strip()])
    search_terms.append(initial_article_data['title']) 
    related_search_query = " ".join(dict.fromkeys(search_terms)) + " latest technology"

    num_additional_needed = max_articles_to_combine - len(all_articles_data)
    if num_additional_needed <= 0: return all_articles_data

    num_to_fetch_candidates = min(num_additional_needed * 2, 5) 
    add_log(f"Searching for up to {num_to_fetch_candidates} related articles with query: \"{related_search_query}\"")
    additional_potential_sites = fetch_news_from_duckduckgo(query=related_search_query, num_results=num_to_fetch_candidates)
    
    related_urls_in_this_batch = {initial_article_data['url']} 

    for site_info_related in additional_potential_sites:
        if stop_event_global.is_set(): break
        if len(all_articles_data) >= max_articles_to_combine: add_log("Reached max articles to combine."); break
        if current_total_words >= min_post_word_count: add_log(f"Reached target word count ({current_total_words})."); break

        related_url = site_info_related['url']
        if related_url in posted_urls_set or related_url in related_urls_in_this_batch:
            add_log(f"Skipping related URL (already posted or in batch): {related_url}")
            continue
        
        related_urls_in_this_batch.add(related_url)
        add_log(f"Attempting to scrape related article: {related_url}")
        if stop_event_global.wait(random.uniform(4, 8)): break # Interruptible sleep
        related_article_data = scrape_article_content(related_url)
        
        if related_article_data and related_article_data.get("text") and related_article_data.get("title"):
            related_word_count = related_article_data.get('word_count',0)
            if related_word_count < 200 : # Min length for related articles
                 add_log(f"Related article '{related_article_data['title']}' too short ({related_word_count} words). Skipping.")
                 continue
            
            all_articles_data.append(related_article_data)
            current_total_words += related_article_data.get('word_count',0)
            add_log(f"Added related: '{related_article_data['title']}' ({related_word_count} words). Total: {current_total_words}")
        else:
            add_log(f"Failed to scrape or insufficient content from related URL: {related_url}")
    
    add_log(f"Finished gathering related. Total articles: {len(all_articles_data)}, Total words: {current_total_words}")
    return all_articles_data


def fetch_related_youtube_video(query):
    global _YOUTUBE_API_KEY_GLOBAL
    if not _YOUTUBE_API_KEY_GLOBAL or _YOUTUBE_API_KEY_GLOBAL == "AIzaSyB7P7wzEaH46OqmC6gs9VY7sqMZzuLSBh4": # Placeholder
        add_log("YouTube API key not set or is placeholder globally. Skipping video search.")
        return None

    add_log(f"Searching YouTube for related video: {query}")
    api_url = "https://youtube.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet", "q": query + " technology review OR explanation",
        "type": "video", "videoEmbeddable": "true", "maxResults": 1,
        "order": "relevance", "key": _YOUTUBE_API_KEY_GLOBAL
    }
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('items'):
            video = data['items'][0]
            add_log(f"Found YouTube video: {video['snippet']['title']}")
            return {'title': video['snippet']['title'], 'videoId': video['id']['videoId']}
        add_log("No related embeddable videos found on YouTube.")
        return None
    except Exception as e:
        add_log(f"Error fetching YouTube data for related video: {e}")
        return None

def format_blog_post_content(list_of_article_data, video_data=None):
    # (Content of this function is identical to original, assuming it's fine)
    # No add_log calls typically needed here as it's formatting.
    if not list_of_article_data: return ""

    primary_article = list_of_article_data[0]
    main_title = primary_article.get('title', "Untitled Tech Article") 
    
    content = f"<h1 style='text-align:center; margin-bottom:25px; font-size:2.2em; color:#2c3e50;'>{main_title}</h1>\n"

    if primary_article['top_image_url']:
        content += f'<div style="text-align:center; margin-bottom:20px;"><img src="{primary_article["top_image_url"]}" alt="{main_title}" style="max-width:90%;height:auto;border-radius:8px;box-shadow: 0 4px 8px rgba(0,0,0,0.1);"></div>\n'
    
    additional_images_html = ""
    added_image_count = 0
    if primary_article.get('all_images'):
        valid_image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
        image_urls_to_display = []
        for img_url in primary_article['all_images']:
            if img_url == primary_article['top_image_url']: continue
            if isinstance(img_url, str) and img_url.lower().endswith(valid_image_extensions):
                 image_urls_to_display.append(img_url)
        
        random.shuffle(image_urls_to_display)

        for img_url in image_urls_to_display:
            if added_image_count >= 2: break # Max 2 additional inline images
            if "icon" in img_url.lower() or "logo" in img_url.lower() or "avatar" in img_url.lower() or "spinner" in img_url.lower():
                continue
            additional_images_html += f'<div style="text-align:center; margin: 15px auto;"><img src="{img_url}" alt="Related image for {main_title}" style="max-width:75%;height:auto;border-radius:6px;box-shadow: 0 2px 4px rgba(0,0,0,0.08);"></div>\n'
            added_image_count += 1
    if additional_images_html:
        content += f"<div class='additional-images-section' style='margin-bottom:25px;'>{additional_images_html}</div>"

    overview_text = ""
    overview_title = "Tech Highlight" 
    if primary_article.get('ai_meta_description') and len(primary_article['ai_meta_description']) > 70 :
        overview_text = primary_article['ai_meta_description']
    elif primary_article.get('summary') and len(primary_article['summary']) > 100:
        overview_text = primary_article['summary']
    else: 
        text_to_use_for_intro = primary_article.get('text', "")
        first_para = text_to_use_for_intro.split('\n')[0].strip()
        if len(first_para) > 100:
            overview_text = first_para
            overview_title = "Introduction"

    if overview_text:
        formatted_overview = ""
        for para_sum in overview_text.split('\n'):
             if para_sum.strip(): formatted_overview += f"<p style='font-size:1.15em; line-height:1.65; color:#34495e;'>{para_sum.strip()}</p>"
        if formatted_overview:
             content += f"<div style='padding:20px; background-color:#f0f4f8; border-left: 5px solid #007bff; margin-bottom:30px; border-radius:5px;'>\n<h3 style='margin-top:0; color:#0056b3; font-size:1.4em;'>{overview_title}</h3>\n{formatted_overview}\n</div>"

    if video_data:
        content += f"""
        <h3 style="margin-top: 30px; margin-bottom:10px; color:#2c3e50; font-size:1.4em;">Related Video: {video_data['title']}</h3>
        <div class="video-container" style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; background: #000; margin-bottom:30px; border-radius:8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
                    src="https://www.youtube.com/embed/{video_data['videoId']}"
                    frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
            </iframe>
        </div>\n"""

    for i, article_data in enumerate(list_of_article_data):
        section_title_text = f"In-Depth Report"
        if len(list_of_article_data) > 1:
            section_title_text += f": Part {i+1}"
            if i == 0 : section_title_text += f" (Primary Analysis)"
            else: section_title_text += f" (Supporting Details)"

        content += f"<h2 style='margin-top: 35px; border-bottom: 3px solid #3498db; padding-bottom: 10px; color:#2980b9; font-size:1.8em;'>{section_title_text}</h2>\n"
        
        if i > 0 and article_data['title'].lower() != main_title.lower():
             content += f"<h4 style='color:#555; margin-bottom:15px; font-style:italic; font-size:1.2em;'>Focus: {article_data['title']}</h4>\n"
        
        current_article_text = article_data['text'] 
        current_article_paragraphs = [p.strip() for p in current_article_text.split('\n') if p.strip() and len(p.strip()) > 15]

        if not current_article_paragraphs:
            content += "<p><em>Detailed textual content for this section was not available or was minimal.</em></p>\n"
        else:
            for para_idx, para in enumerate(current_article_paragraphs):
                style = "line-height:1.75; font-size:1.1em; margin-bottom:1.2em; color:#34495e;"
                if para_idx == 0 and len(para) > 70 and i == 0 : # Drop cap for first para of main article
                    first_char = para[0]
                    rest_of_para = para[1:]
                    content += f'<p style="{style}"><span style="float:left; font-size:3.5em; line-height:0.8em; margin-right:0.07em; margin-top:0.1em; font-weight:bold; color:#3498db;">{first_char}</span>{rest_of_para}</p>\n'
                else:
                    content += f"<p style='{style}'>{para}</p>\n"
        
        meta_html = ""
        if article_data['authors']: meta_html += f"<strong>Author(s) (Original Source):</strong> {', '.join(article_data['authors'])}<br>"
        if article_data['publish_date'] and article_data['publish_date'] != "N/A": meta_html += f"<strong>Original Publish Date:</strong> {article_data['publish_date']}<br>"
        
        if meta_html:
            content += f"<div style='font-size:0.9em; color:#7f8c8d; margin-top:20px; padding:12px; border:1px solid #ecf0f1; background-color:#fdfefe; border-radius:4px;'>{meta_html}</div>\n"

    total_words_in_post = sum(ad.get('word_count',0) for ad in list_of_article_data)
    content += f"<hr style='margin-top:40px; margin-bottom:20px; border:0; border-top:1px solid #bdc3c7;'>"
    content += f"<p style='font-size:0.9em; color:#95a5a6; text-align:center;'><em>Disclaimer: This content is automatically curated and significantly enhanced by AI for thematic focus and readability. It is compiled from various online news reports for comprehensiveness. While efforts are made for accuracy, information can change rapidly. This post contains approximately {total_words_in_post} words.</em></p>"
    
    return content


# --- Google OAuth and Blogger API Functions ---
def get_credentials(client_id, client_secret):
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            add_log(f"Error loading token from file: {e}. Will try to re-authenticate.")
            creds = None
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                add_log("Credentials expired, attempting refresh...")
                creds.refresh(Request())
                with open(TOKEN_FILE, 'w') as token_f:
                    token_f.write(creds.to_json())
                add_log("Credentials refreshed successfully.")
                return creds
            except Exception as e:
                add_log(f"Failed to refresh credentials: {e}. Need to re-authenticate.")
                if os.path.exists(TOKEN_FILE): # Remove potentially corrupt token file
                    try: os.remove(TOKEN_FILE)
                    except: pass
                creds = None # Force re-authentication
        
        if not creds: # Needs full authentication
            add_log("No valid credentials, starting authentication flow...")
            # Construct client_config for InstalledAppFlow
            client_config_dict = {
                "installed": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost", "urn:ietf:wg:oauth:2.0:oob"]
                }
            }
            try:
                flow = InstalledAppFlow.from_client_config(client_config_dict, SCOPES)
                # run_local_server is generally preferred for better UX if browser access is available
                # It opens a browser and listens on a local port.
                # port=0 lets it pick an available port.
                add_log("Attempting to launch browser for authentication (InstalledAppFlow)...")
                creds = flow.run_local_server(port=0) 
                # If run_local_server fails (e.g. no GUI browser), run_console() can be a fallback
                # but requires manual copy-pasting of URL and code.
            except Exception as e_local_server:
                add_log(f"run_local_server failed: {e_local_server}. Trying run_console as fallback.")
                try:
                    flow = InstalledAppFlow.from_client_config(client_config_dict, SCOPES)
                    creds = flow.run_console()
                except Exception as e_console:
                    add_log(f"run_console authentication flow also failed: {e_console}")
                    return None

            if creds:
                with open(TOKEN_FILE, 'w') as token_f:
                    token_f.write(creds.to_json())
                add_log("Authentication successful. Credentials saved.")
            else:
                add_log("Authentication flow did not return credentials.")
    
    if creds and creds.valid:
        return creds
    else:
        add_log("Failed to obtain valid credentials after all attempts.")
        return None


def get_blogger_service_instance(credentials):
    if not credentials or not credentials.valid:
        add_log("Credentials invalid or missing when trying to build Blogger service.")
        return None
    try:
        service = build('blogger', 'v3', credentials=credentials, cache_discovery=False)
        return service
    except Exception as e:
        add_log(f"Error building Blogger service: {e}")
        return None

def post_to_blogger_oauth(credentials_obj, current_blog_id_val, article_title, content_html, labels_list):
    if not credentials_obj:
        add_log("ERROR: Blogger credentials not available for posting.")
        return None

    service = get_blogger_service_instance(credentials_obj)
    if not service:
        add_log("Failed to initialize Blogger service. Cannot post.")
        # Potentially signal for re-authentication if this happens often
        st.session_state.credentials = None # Invalidate creds to force re-auth
        if os.path.exists(TOKEN_FILE): 
            try: os.remove(TOKEN_FILE)
            except: pass
        return None

    final_labels = ['Tech News', 'AI Update', 'Gadget Review', 'Automated Post']
    for lbl in labels_list:
        cleaned_lbl = lbl.strip()
        if cleaned_lbl and cleaned_lbl not in final_labels and len(final_labels) < 20:
            final_labels.append(cleaned_lbl)
    
    post_body = {'kind': 'blogger#post', 'title': article_title, 'content': content_html, 'labels': final_labels}
    
    try:
        add_log(f"Attempting to post to Blogger Blog ID {current_blog_id_val}: \"{article_title}\" with {len(final_labels)} labels.")
        posts_api = service.posts()
        request = posts_api.insert(blogId=current_blog_id_val, body=post_body, isDraft=False)
        response = request.execute()
        add_log(f"Successfully posted to Blogger: {article_title} (ID: {response.get('id')})")
        return response
    except HttpError as e:
        error_content = e.resp.reason
        try:
            error_details = json.loads(e.content).get('error', {})
            error_message_detail = error_details.get('message', e.resp.reason)
            error_status = error_details.get('status', 'UNKNOWN_STATUS')
            add_log(f"Error posting to Blogger (HttpError {e.resp.status} - {error_status}): {error_message_detail}")
        except: # Fallback if content is not JSON
             add_log(f"Error posting to Blogger (HttpError {e.resp.status}): {e.resp.reason}")

        if e.resp.status == 401 or "invalid_grant" in str(e).lower() or "token has been expired or revoked" in str(e).lower():
            add_log("Blogger API: Access token likely expired or revoked. Invalidating local token for re-authentication.")
            st.session_state.credentials = None # This will prompt re-auth in UI
            if os.path.exists(TOKEN_FILE):
                try: os.remove(TOKEN_FILE)
                except Exception as ex_del: add_log(f"Could not delete token file: {ex_del}")
        return None
    except Exception as e:
        add_log(f"Generic error posting to Blogger: {e}")
        return None

def check_blogger_token_validity(credentials_obj, current_blog_id_val):
    if not credentials_obj:
        add_log("Credentials not available for token validity check.")
        return False
    
    service = get_blogger_service_instance(credentials_obj)
    if not service:
        add_log("Failed to initialize Blogger service for token check.")
        return False

    add_log(f"Checking Blogger token validity for Blog ID: {current_blog_id_val}...")
    try:
        service.blogs().get(blogId=current_blog_id_val).execute()
        add_log("Access token is valid for Blogger.")
        return True
    except HttpError as e:
        add_log(f"Blogger token check failed (HttpError {e.resp.status}): {e.resp.reason}")
        if e.resp.status == 401:
             add_log("Token seems invalid or expired. Invalidating for re-auth.")
             st.session_state.credentials = None # This will prompt re-auth in UI
             if os.path.exists(TOKEN_FILE):
                try: os.remove(TOKEN_FILE)
                except Exception as ex_del: add_log(f"Could not delete token file: {ex_del}")
        return False
    except Exception as e:
        add_log(f"Error during Blogger token validity check: {e}")
        return False

# --- Bot Thread Function ---
def bot_worker_thread(stop_event, config):
    global _YOUTUBE_API_KEY_GLOBAL, _GEMINI_API_KEY_GLOBAL, _BLOG_ID_GLOBAL

    # Set global API keys from config for helper functions
    _YOUTUBE_API_KEY_GLOBAL = config['youtube_api_key']
    _BLOG_ID_GLOBAL = config['blog_id']
    
    # Configure Gemini API at the start of the thread
    if not configure_gemini_api_runtime(config['gemini_api_key']):
        add_log("Gemini API not configured. AI Enhancement will be skipped in bot thread.")
        # Depending on strictness, could stop if Gemini is critical:
        # add_log("Exiting bot thread as Gemini API is critical and not configured.")
        # return

    add_log(f"Bot thread started. Target: >{config['min_post_word_count']} words. Interval: {config['post_interval_minutes']} min.")
    download_nltk_resources()

    # Credentials should be passed and managed by the main Streamlit app state.
    # The bot thread will use the credentials object it's given.
    # If it becomes invalid, post_to_blogger_oauth or check_token_validity should handle it
    # by setting st.session_state.credentials to None, which the UI will pick up.

    posted_urls = load_posted_urls()
    add_log(f"Loaded {len(posted_urls)} previously posted URLs from {POSTED_URLS_FILE}.")

    error_count = 0
    max_errors_in_row = 7 # Max consecutive errors before pausing/stopping bot
    search_queries = [
        "latest artificial intelligence breakthroughs", "new consumer electronics and gadgets", 
        "advances in machine learning research", "innovative software development trends",
        "cutting-edge robotics technology news", "future of tech detailed analysis",
        "semiconductor industry news and chips", "virtual reality and augmented reality updates",
        "next generation AI models and applications", "wearable technology reviews and news",
        "space technology and commercial spaceflight", "cybersecurity threats and AI defense"
    ]
    current_query_index = 0

    while not stop_event.is_set():
        try:
            # --- Check Credentials before each major operation ---
            # The credentials object is managed by st.session_state in the main thread.
            # The bot thread reads this state at the start of each iteration.
            # This is a simplification; for true shared state, a lock might be needed if written from here.
            # However, if creds become invalid, functions like post_to_blogger will nullify them in st.session_state.
            current_credentials_from_st = st.session_state.get('credentials')
            if not current_credentials_from_st or not current_credentials_from_st.valid:
                add_log("Credentials invalid or missing at start of bot iteration. Pausing. Please re-authenticate via UI.")
                if stop_event.wait(60): break # Wait and check stop event
                continue # Skip this iteration, wait for user to re-auth

            search_query = search_queries[current_query_index % len(search_queries)]
            current_query_index += 1
            add_log(f"--- Bot Iteration: Query '{search_query}' ---")
            
            potential_sites_from_ddg = fetch_news_from_duckduckgo(query=search_query, num_results=3)
            posted_article_in_this_iteration = False
            
            for site_info in potential_sites_from_ddg:
                if stop_event.is_set(): break
                if posted_article_in_this_iteration: break

                ddg_url = site_info['url']
                ddg_title = site_info['title']
                add_log(f"Processing DDG result: '{ddg_title}' ({ddg_url})")

                primary_article_data = None
                if ddg_url not in posted_urls:
                    add_log(f"Attempting direct scrape: {ddg_url}")
                    if stop_event.wait(random.uniform(1,3)): break
                    primary_article_data = scrape_article_content(ddg_url)
                    if not primary_article_data:
                        add_log(f"Direct scrape failed or insufficient content for {ddg_url}.")
                else:
                    add_log(f"DDG URL {ddg_url} already processed. Skipping direct.")
                
                if not primary_article_data: # Try building from source if direct fails or homepage-like
                    parsed_uri = requests.utils.urlparse(ddg_url)
                    is_likely_homepage = not parsed_uri.path or parsed_uri.path == '/' or len(parsed_uri.path.split('/')) <= 2
                    if is_likely_homepage or (not primary_article_data and ddg_url not in posted_urls):
                        add_log(f"Treating {ddg_url} as news source page for sub-articles...")
                        try:
                            source_config = newspaper.Config()
                            source_config.request_timeout = 30; source_config.browser_user_agent = get_random_user_agent()
                            source_config.fetch_images = False; source_config.memoize_articles = False
                            source = newspaper.build(ddg_url, config=source_config)
                            add_log(f"Found {len(source.articles)} links on {ddg_url}. Processing up to 2 tech-focused.")
                            
                            processed_from_source = 0
                            for sub_article_obj in source.articles:
                                if stop_event.is_set() or processed_from_source >= 2 or posted_article_in_this_iteration: break
                                sub_url = sub_article_obj.url
                                if not sub_url or not sub_url.startswith('http') or sub_url in posted_urls: continue
                                
                                add_log(f"Scraping sub-article: {sub_url}")
                                if stop_event.wait(random.uniform(2,5)): break
                                scraped_sub = scrape_article_content(sub_url)
                                processed_from_source += 1
                                if scraped_sub:
                                    primary_article_data = scraped_sub
                                    add_log(f"Obtained primary from sub-link: {primary_article_data['title']}")
                                    break 
                        except Exception as e_build: add_log(f"Error building/processing source {ddg_url}: {e_build}")
                
                if stop_event.is_set(): break
                
                if primary_article_data:
                    if primary_article_data['url'] in posted_urls:
                        add_log(f"Article {primary_article_data['url']} already posted. Skipping."); continue

                    add_log(f"Primary article: '{primary_article_data['title']}' ({primary_article_data.get('word_count',0)} words)")
                    
                    if _GEMINI_API_KEY_GLOBAL: # Check global config for Gemini
                        primary_article_data = enhance_article_with_ai(primary_article_data)
                    else:
                        add_log("Skipping AI enhancement (Gemini key not configured globally).")
                    
                    list_of_articles = get_sufficient_article_content(primary_article_data, posted_urls, 
                                                                    config['min_post_word_count'], config['max_articles_to_combine'])
                    final_words = sum(ad.get('word_count',0) for ad in list_of_articles)
                    
                    if final_words >= config['min_post_word_count']:
                        add_log(f"Content meets target: {final_words} words from {len(list_of_articles)} article(s).")
                        post_title = list_of_articles[0]['title']
                        keywords = set(list_of_articles[0].get('ai_keywords', []))
                        for ad in list_of_articles: keywords.update(kw.strip().capitalize() for kw in ad.get('keywords', []) if kw.strip())

                        vid_query = list_of_articles[0]['title'] 
                        if list_of_articles[0].get('ai_keywords'): vid_query += " " + " ".join(list_of_articles[0]['ai_keywords'][:2])
                        video = fetch_related_youtube_video(vid_query)
                        
                        html_content = format_blog_post_content(list_of_articles, video)
                        
                        # Use current credentials from Streamlit session state for posting
                        if post_to_blogger_oauth(st.session_state.get('credentials'), _BLOG_ID_GLOBAL, post_title, html_content, list(keywords)):
                            for ad in list_of_articles:
                                if ad['url'] not in posted_urls: save_posted_url(ad['url']); posted_urls.add(ad['url'])
                            posted_article_in_this_iteration = True
                            error_count = 0 # Reset error count on success
                            break # Successfully posted, move to next main loop iteration after wait
                    else:
                        add_log(f"Combined content ({final_words} words) less than target. Skipping post.")
            
            if stop_event.is_set(): add_log("Stop event detected, exiting bot loop."); break

            if posted_article_in_this_iteration:
                add_log("Successfully posted an article in this iteration.")
                wait_duration = config['post_interval_minutes'] * 60
            else:
                add_log("No new articles posted in this iteration.")
                wait_duration = min(config['post_interval_minutes'] * 60, 15 * 60) # Shorter wait if no post, capped by interval

            add_log(f"Waiting for {wait_duration // 60}m {wait_duration % 60}s...")
            if stop_event.wait(wait_duration): add_log("Wait interrupted."); break
        
        except Exception as e_loop:
            add_log(f"CRITICAL ERROR IN BOT MAIN LOOP: {e_loop}")
            str_io = io.StringIO()
            tb_module.print_exc(file=str_io)
            add_log(str_io.getvalue())
            str_io.close()
            error_count += 1
            if error_count >= max_errors_in_row:
                add_log(f"Max ({max_errors_in_row}) critical errors reached. Stopping bot thread."); break
            add_log(f"Error count: {error_count}/{max_errors_in_row}. Waiting 5 mins before retrying...")
            if stop_event.wait(300): break

    add_log("Bot thread finished.")
    # Signal main thread that bot has stopped (if not by explicit stop button)
    # This requires care; main thread should check thread.is_alive() or a flag.
    # For now, relying on UI stop button and stop_event.
    if 'bot_running' in st.session_state and st.session_state.bot_running:
        # This is tricky because st.session_state should ideally be modified by the main thread.
        # A more robust solution might involve another queue or callback for the main thread to update UI state.
        # For simplicity, we'll let the main thread's periodic refresh handle UI update if bot dies.
        add_log("Bot stopped unexpectedly or completed. UI may need refresh to update status.")


# --- Streamlit App ---
def streamlit_main():
    st.set_page_config(layout="wide")
    st.title("🤖 AI Tech News to Blogger Auto-Poster")

    # --- Initialize Streamlit Session State ---
    if 'bot_running' not in st.session_state: st.session_state.bot_running = False
    if 'credentials' not in st.session_state: st.session_state.credentials = None
    if 'logs' not in st.session_state: st.session_state.logs = []
    if 'bot_thread' not in st.session_state: st.session_state.bot_thread = None
    
    # --- Sidebar for Configuration and Control ---
    st.sidebar.header("⚙️ Configuration")
    client_id = st.sidebar.text_input("Google Client ID", value="1082055366763-9a1e944hf42vfg6ca9dao52oij4sis43.apps.googleusercontent.com", help="Your Google OAuth Client ID")
    client_secret = st.sidebar.text_input("Google Client Secret", value="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", type="password", help="Your Google OAuth Client Secret")
    
    st.session_state.blog_id_input = st.sidebar.text_input("Blogger Blog ID", value=st.session_state.get('blog_id_input', "1442443742033959520"), help="The ID of your Blogger blog")
    st.session_state.youtube_api_key_input = st.sidebar.text_input("YouTube API Key", value=st.session_state.get('youtube_api_key_input', "AIzaSyB7P7wzEaH46OqmC6gs9VY7sqMZzuLSBh4"), help="Needed for finding related videos")
    st.session_state.gemini_api_key_input = st.sidebar.text_input("Gemini API Key", value=st.session_state.get('gemini_api_key_input', "AIzaSyCrsJE4ZR_RxBvy2rlGjWalVkuUudKTm0c"), help="Needed for AI content enhancement")
    
    post_interval_min = st.sidebar.number_input("Post Interval (minutes)", min_value=5, value=st.session_state.get('post_interval_min', 10), step=1, help="How often to attempt a new post")
    min_word_count = st.sidebar.number_input("Min Post Word Count", min_value=500, value=DEFAULT_MIN_POST_WORD_COUNT, step=100)
    max_articles_combine = st.sidebar.number_input("Max Articles to Combine", min_value=1, max_value=5, value=DEFAULT_MAX_ARTICLES_TO_COMBINE, step=1)

    st.sidebar.header("🔒 Authentication")
    # Load credentials from file if they exist and are not in session state yet
    if not st.session_state.credentials and os.path.exists(TOKEN_FILE):
        try:
            st.session_state.credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            if st.session_state.credentials and st.session_state.credentials.expired and st.session_state.credentials.refresh_token:
                st.session_state.credentials.refresh(Request()) # Refresh if expired
                with open(TOKEN_FILE, 'w') as token_f: token_f.write(st.session_state.credentials.to_json())
            if st.session_state.credentials and st.session_state.credentials.valid:
                 add_log("Loaded and validated credentials from token.json.")
            else: # Failed to load or validate
                st.session_state.credentials = None
                if os.path.exists(TOKEN_FILE): os.remove(TOKEN_FILE) # Clean up bad token file
        except Exception as e:
            add_log(f"Error auto-loading token: {e}. Please authenticate manually.")
            st.session_state.credentials = None
            if os.path.exists(TOKEN_FILE): os.remove(TOKEN_FILE)


    if st.session_state.credentials and st.session_state.credentials.valid:
        st.sidebar.success("Authenticated with Google ✔️")
        # Check token validity for the specific blog ID (can be a bit slow, do it less often or on demand)
        # if st.sidebar.button("Verify Token for Blog"):
        #    check_blogger_token_validity(st.session_state.credentials, st.session_state.blog_id_input)

        if st.sidebar.button("Logout / Clear Credentials"):
            st.session_state.credentials = None
            if os.path.exists(TOKEN_FILE):
                try: os.remove(TOKEN_FILE)
                except Exception as e: add_log(f"Error removing token file: {e}")
            add_log("Credentials cleared.")
            st.rerun()
    else:
        st.sidebar.warning("Not Authenticated.")
        if st.sidebar.button("Authenticate with Google"):
            if not client_id or not client_secret:
                st.sidebar.error("Client ID and Client Secret must be provided.")
            else:
                with st.spinner("Attempting Google Authentication... Check console/browser if needed."):
                    creds = get_credentials(client_id, client_secret)
                if creds and creds.valid:
                    st.session_state.credentials = creds
                    st.sidebar.success("Authentication Successful!")
                    add_log("Google Authentication successful via UI button.")
                    st.rerun()
                else:
                    st.sidebar.error("Authentication Failed. Check logs.")
                    add_log("Google Authentication failed via UI button.")

    # --- Bot Control Area ---
    st.header("▶️ Bot Control")
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.bot_running:
            if st.button("⏹️ Stop Bot", type="primary"):
                add_log("Stop button clicked. Signaling bot to stop...")
                stop_event_global.set()
                if st.session_state.bot_thread and st.session_state.bot_thread.is_alive():
                    st.session_state.bot_thread.join(timeout=10) # Wait for thread to finish
                    if st.session_state.bot_thread.is_alive():
                        add_log("Bot thread did not stop in time.")
                    else:
                        add_log("Bot thread stopped successfully.")
                st.session_state.bot_running = False
                st.session_state.bot_thread = None
                st.rerun()
        else:
            if st.button("🚀 Start Bot"):
                if not st.session_state.credentials or not st.session_state.credentials.valid:
                    st.error("Please authenticate with Google first.")
                elif not st.session_state.blog_id_input:
                    st.error("Blogger Blog ID is required.")
                elif not st.session_state.youtube_api_key_input or st.session_state.youtube_api_key_input == "AIzaSyB7P7wzEaH46OqmC6gs9VY7sqMZzuLSBh4": # Placeholder
                    st.error("Valid YouTube API Key is required.")
                elif not st.session_state.gemini_api_key_input or st.session_state.gemini_api_key_input == "AIzaSyCrsJE4ZR_RxBvy2rlGjWalVkuUudKTm0c": # Placeholder
                    st.error("Valid Gemini API Key is required.")
                else:
                    add_log("Start button clicked. Starting bot thread...")
                    stop_event_global.clear()
                    st.session_state.bot_running = True
                    
                    bot_config = {
                        'youtube_api_key': st.session_state.youtube_api_key_input,
                        'gemini_api_key': st.session_state.gemini_api_key_input,
                        'blog_id': st.session_state.blog_id_input,
                        'post_interval_minutes': post_interval_min,
                        'min_post_word_count': min_word_count,
                        'max_articles_to_combine': max_articles_combine,
                    }
                    
                    thread = threading.Thread(target=bot_worker_thread, args=(stop_event_global, bot_config), daemon=True)
                    st.session_state.bot_thread = thread
                    thread.start()
                    st.rerun()
    
    with col2:
        status_color = "green" if st.session_state.bot_running else "red"
        status_text = "RUNNING" if st.session_state.bot_running else "STOPPED"
        st.markdown(f"**Bot Status:** <span style='color:{status_color};'>{status_text}</span>", unsafe_allow_html=True)
        if st.session_state.bot_thread and not st.session_state.bot_thread.is_alive() and st.session_state.bot_running:
             # Bot was running but thread died
             st.error("Bot thread seems to have stopped unexpectedly! Check logs.")
             st.session_state.bot_running = False # Correct the state


    # --- Log Display Area ---
    st.header("📜 Bot Activity Logs")
    log_placeholder = st.empty()
    
    # Process log queue
    new_logs = []
    while not log_queue_global.empty():
        try:
            new_logs.append(log_queue_global.get_nowait())
        except queue.Empty:
            break
    
    if new_logs:
        st.session_state.logs = new_logs + st.session_state.logs # Prepend new logs
        if len(st.session_state.logs) > 200: # Limit log history
            st.session_state.logs = st.session_state.logs[:200]

    log_placeholder.text_area("", value="\n".join(st.session_state.logs), height=400, key="log_display_area_unique")

    # Auto-refresh logs if bot is running
    if st.session_state.bot_running:
        time.sleep(2) # Refresh interval for logs (seconds)
        st.rerun()

if __name__ == "__main__":
    streamlit_main()
