import os
import json
import time
import re
import logging
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from rapidfuzz import fuzz
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Environment Configuration
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://greenxonline.com")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SHOPIFY_CATALOG_URL = os.getenv("SHOPIFY_CATALOG_URL", "https://greenxonline.com/products.json")
FAWA_MASTER_KEY = os.getenv("FAWA_MASTER_KEY", "fawa")
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB limit
CATALOG_TTL_SECONDS = 1800  # 30 minutes in-memory cache

# Configure CORS restricted to ALLOWED_ORIGIN and localhost for development
origins = [ALLOWED_ORIGIN]
if ALLOWED_ORIGIN != "*":
    origins.extend(["http://localhost:3000", "http://localhost:5000", "http://127.0.0.1:5000", "http://127.0.0.1:3000"])

CORS(app, resources={r"/api/*": {"origins": origins}})

# Rate limiting per IP (20 requests per hour for plant identification)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

CATALOG_BACKUP_FILE = os.path.join(os.path.dirname(__file__), "catalog_backup.json")


def load_disk_catalog_backup():
    """Loads catalog from local disk backup if Shopify HTTP request fails or rate limits."""
    if os.path.exists(CATALOG_BACKUP_FILE):
        try:
            with open(CATALOG_BACKUP_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    logger.info(f"Loaded {len(data)} products from local disk catalog backup.")
                    return data
        except Exception as e:
            logger.error(f"Failed to read disk catalog backup: {e}")
    return []


# Global in-memory catalog cache pre-loaded with bundled catalog backup
_initial_products = load_disk_catalog_backup()
_catalog_cache = {
    "timestamp": time.time() if _initial_products else 0,
    "products": _initial_products
}

FAMILY_GENERA_MAP = {
    "araceae": ["anthurium", "monstera", "philodendron", "alocasia", "syngonium", "aglaonema", "caladium", "spathiphyllum", "zamioculcas", "dieffenbachia", "scindapsus", "epipremnum", "rhaphidophora", "homalomena", "aroid"],
    "arecaceae": ["chamaedorea", "dypsis", "livistona", "rhapis", "phoenix", "howea", "palm"],
    "asparagaceae": ["sansevieria", "dracaena", "aspidistra", "chlorophytum", "cordyline", "beaucarnea", "yucca", "agave", "snake plant"],
    "cactaceae": ["opuntia", "echinocactus", "mammillaria", "rhipsalis", "epiphyllum", "cactus", "succulent"],
    "moraceae": ["ficus", "fig", "rubber plant"],
    "piperaceae": ["peperomia", "piper"],
    "marantaceae": ["calathea", "maranta", "ctenanthe", "stromanthe", "prayer plant"],
    "crassulaceae": ["echeveria", "crassula", "sedum", "kalanchoe", "sempervivum", "jade plant"],
    "bromeliaceae": ["tillandsia", "guzmania", "vriesea", "aechmea", "neoregelia"]
}


def save_disk_catalog_backup(products):
    """Saves fetched catalog to local disk backup."""
    if products and isinstance(products, list):
        try:
            with open(CATALOG_BACKUP_FILE, "w", encoding="utf-8") as f:
                json.dump(products, f)
            logger.info(f"Saved {len(products)} products to local disk catalog backup.")
        except Exception as e:
            logger.error(f"Failed to write disk catalog backup: {e}")


def get_shopify_catalog(force_refresh=False):
    """
    Fetch full product catalog from Shopify store (/products.json) with pagination,
    custom headers to avoid 429 rate limiting, and persistent disk backup fallback.
    """
    global _catalog_cache
    now = time.time()
    
    if not force_refresh and _catalog_cache["products"] and (now - _catalog_cache["timestamp"] < CATALOG_TTL_SECONDS):
        logger.info(f"Returning cached catalog with {len(_catalog_cache['products'])} products.")
        return _catalog_cache["products"]

    logger.info(f"Fetching fresh product catalog from {SHOPIFY_CATALOG_URL}...")
    products = []
    page = 1
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    try:
        while True:
            separator = "&" if "?" in SHOPIFY_CATALOG_URL else "?"
            # Use lower limit if initial request gets throttled
            limit_param = 250 if page == 1 else 100
            url = f"{SHOPIFY_CATALOG_URL}{separator}limit={limit_param}&page={page}"
            resp = requests.get(url, headers=headers, timeout=12)
            
            if resp.status_code == 429:
                logger.warning(f"Shopify catalog fetch received 429 rate limit on page {page}. Retrying after 1.5s delay...")
                time.sleep(1.5)
                # Retry with smaller page limit
                url_retry = f"{SHOPIFY_CATALOG_URL}{separator}limit=50&page={page}"
                resp = requests.get(url_retry, headers=headers, timeout=12)
            
            if resp.status_code != 200:
                logger.error(f"Shopify catalog fetch failed on page {page} with status {resp.status_code}")
                break
            
            data = resp.json()
            page_products = data.get("products", [])
            if not page_products:
                break
                
            products.extend(page_products)
            logger.info(f"Page {page}: fetched {len(page_products)} products (Total: {len(products)})")
            
            if len(page_products) < limit_param:
                break
            page += 1
            
        if products:
            _catalog_cache["timestamp"] = now
            _catalog_cache["products"] = products
            save_disk_catalog_backup(products)
            logger.info(f"Successfully cached {len(products)} catalog products.")
            return products

    except Exception as e:
        logger.error(f"Error fetching catalog from Shopify: {e}")

    # Fallbacks if HTTP request failed or was rate limited
    if _catalog_cache["products"]:
        logger.info("Serving in-memory catalog cache due to fetch error.")
        return _catalog_cache["products"]

    disk_backup = load_disk_catalog_backup()
    if disk_backup:
        _catalog_cache["products"] = disk_backup
        _catalog_cache["timestamp"] = now
        return disk_backup

    return products


def clean_json_response(raw_text):
    """
    Strips markdown code fences (e.g. ```json ... ```) and cleans LLM response for JSON parsing.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def call_gemini_vision(image_bytes, mime_type):
    """
    Sends customer photo to Gemini Vision model to identify plant common names, scientific name,
    genus, family name, family keywords, and visual traits.
    Supports multi-model fallbacks for 503 High Demand errors.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    prompt = (
        "You are identifying a houseplant from a customer photo for a plant store's visual search feature.\n"
        "Analyze the plant in this photo and respond with ONLY valid JSON, no markdown fences, no preamble, in this exact JSON format:\n\n"
        "{\n"
        '  "common_names": ["most likely common name", "alternate common name"],\n'
        '  "scientific_name": "Genus species",\n'
        '  "genus": "Genus name (e.g. Anthurium)",\n'
        '  "family_name": "Family name (e.g. Araceae)",\n'
        '  "family_keywords": ["Anthurium", "Araceae", "velvet cardboard leaf", "aroid"],\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "visual_traits": "one short sentence describing leaf shape, color pattern, growth habit"\n'
        "}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. List 1-3 common_names ordered by likelihood.\n"
        "2. Identify the scientific_name, genus, and family_name of the plant accurately.\n"
        "3. In family_keywords, include family_name, genus, common family names, and related plant terms to find all site products belonging to the same plant family.\n"
        "4. If you cannot identify the plant with high confidence, set confidence to 'low' and still give your best guess — never refuse to answer."
    )

    models_to_try = [
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-2.5-flash"
    ]
    
    # Remove duplicates while maintaining order
    unique_models = []
    for m in models_to_try:
        if m and m not in unique_models:
            unique_models.append(m)

    raw_text = ""
    last_exception = None

    for model_name in unique_models:
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt
                ]
            )
            raw_text = response.text
            if raw_text:
                break
        except Exception as err_new_sdk:
            logger.warning(f"google.genai with model {model_name} failed ({err_new_sdk}), trying old SDK / fallback model...")
            try:
                import google.generativeai as genai_old
                genai_old.configure(api_key=api_key)
                model = genai_old.GenerativeModel(model_name)
                
                image_part = {"mime_type": mime_type, "data": image_bytes}
                response = model.generate_content([image_part, prompt])
                raw_text = response.text
                if raw_text:
                    break
            except Exception as err_old_sdk:
                last_exception = err_old_sdk
                continue

    if not raw_text:
        if last_exception:
            raise last_exception
        raise ValueError("Could not get response from Gemini Vision models")

    cleaned = clean_json_response(raw_text)
    try:
        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError as jde:
        logger.error(f"Failed to parse Gemini response as JSON: {cleaned}")
        raise ValueError(f"Invalid JSON returned by Gemini Vision: {jde}")


def shortlist_family_products(ident_result, catalog):
    """
    Shortlists all products from the site catalog belonging to the same plant family/genus.
    Uses FAMILY_GENERA_MAP to expand family terms and ensure all family products are included.
    """
    family_name = ident_result.get("family_name", "").strip().lower()
    genus = ident_result.get("genus", "").strip().lower()
    scientific_name = ident_result.get("scientific_name", "").strip().lower()
    common_names = ident_result.get("common_names", [])
    family_keywords = ident_result.get("family_keywords", [])

    # Build search terms for family shortlisting
    family_terms = set()
    if family_name:
        family_terms.add(family_name)
        # Check map for expanded family genera
        if family_name in FAMILY_GENERA_MAP:
            for g in FAMILY_GENERA_MAP[family_name]:
                family_terms.add(g)

    if genus:
        family_terms.add(genus)

    if scientific_name:
        sp_parts = scientific_name.split()
        if sp_parts:
            family_terms.add(sp_parts[0])

    for kw in family_keywords:
        if kw:
            family_terms.add(str(kw).strip().lower())
    for cn in common_names:
        if cn:
            family_terms.add(str(cn).strip().lower())

    shortlisted = []

    for product in catalog:
        title = product.get("title", "").lower()
        tags = product.get("tags", [])
        if isinstance(tags, list):
            tags_str = " ".join(tags).lower()
        else:
            tags_str = str(tags).lower()
        product_type = product.get("product_type", "").lower()
        handle = product.get("handle", "").lower()

        search_blob = f"{title} {tags_str} {product_type} {handle}"

        family_score = 0.0
        for term in family_terms:
            if not term or len(term) < 3:
                continue
            if term in search_blob:
                if term in title:
                    family_score = max(family_score, 100.0)
                elif term in tags_str:
                    family_score = max(family_score, 90.0)
                else:
                    family_score = max(family_score, 80.0)
            else:
                token_ratio = fuzz.token_set_ratio(term, search_blob)
                partial_ratio = fuzz.partial_ratio(term, search_blob)
                f_ratio = max(token_ratio, partial_ratio)
                if f_ratio >= 60:
                    family_score = max(family_score, float(f_ratio))

        if family_score >= 40:
            shortlisted.append((product, family_score))

    # If no strict family matches found, include top catalog products so crosscheck is never empty
    if not shortlisted:
        logger.info("No strict family matches found; including catalog products for visual crosscheck.")
        shortlisted = [(product, 50.0) for product in catalog]

    shortlisted.sort(key=lambda x: x[1], reverse=True)
    return shortlisted


def evaluate_visual_similarity(user_image_bytes, user_mime_type, candidate_products):
    """
    Downloads candidate product images and calls Gemini Vision to crosscheck & score visual similarity (0-100)
    between the user's uploaded plant photo and candidate product images.
    Returns dict mapping product_id -> visual_score (0-100).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not user_image_bytes:
        return {}

    visual_scores = {}
    valid_candidates = []

    # Download images for candidate products (up to 15 candidates for visual crosscheck)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    for prod in candidate_products[:15]:
        prod_id = prod.get("id")
        images = prod.get("images", [])
        if not images:
            continue
        img_url = images[0].get("src")
        if not img_url:
            continue
        
        try:
            resp = requests.get(img_url, headers=headers, timeout=2.5)
            if resp.status_code == 200 and resp.content:
                img_mime = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
                if not img_mime.startswith("image/"):
                    img_mime = "image/jpeg"
                valid_candidates.append({
                    "id": prod_id,
                    "title": prod.get("title", ""),
                    "bytes": resp.content,
                    "mime": img_mime
                })
        except Exception as err:
            logger.warning(f"Could not download image for candidate product {prod_id}: {err}")
            continue

    if not valid_candidates:
        return {}

    prompt = (
        "Image 0 is a customer's uploaded plant photo. "
        "The following images are candidate product photos of plants belonging to the same family from our store.\n"
        "Crosscheck each candidate product image against Image 0 and rate its visual similarity from 0 to 100 "
        "based on leaf shape, pattern, venation, color, and visual foliage similarity.\n\n"
        "Respond ONLY with a valid JSON object mapping 1-based candidate index to score (0-100):\n"
        "{\n"
        '  "scores": [\n'
        '    {"candidate_index": 1, "visual_score": 85},\n'
        '    {"candidate_index": 2, "visual_score": 60}\n'
        "  ]\n"
        "}"
    )

    try:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=api_key)
            contents = [types.Part.from_bytes(data=user_image_bytes, mime_type=user_mime_type)]
            for cand in valid_candidates:
                contents.append(types.Part.from_bytes(data=cand["bytes"], mime_type=cand["mime"]))
            contents.append(prompt)
            
            response = client.models.generate_content(model=model_name, contents=contents)
            raw_text = response.text
        except Exception as err_new:
            logger.warning(f"google.genai batch visual evaluation failed ({err_new}), falling back to google.generativeai...")
            import google.generativeai as genai_old
            genai_old.configure(api_key=api_key)
            model = genai_old.GenerativeModel(model_name)
            
            contents = [{"mime_type": user_mime_type, "data": user_image_bytes}]
            for cand in valid_candidates:
                contents.append({"mime_type": cand["mime"], "data": cand["bytes"]})
            contents.append(prompt)
            
            response = model.generate_content(contents)
            raw_text = response.text

        cleaned = clean_json_response(raw_text)
        data = json.loads(cleaned)
        scores_list = data.get("scores", [])
        for item in scores_list:
            idx = item.get("candidate_index", 0) - 1
            score = item.get("visual_score", 0)
            if 0 <= idx < len(valid_candidates):
                cand_id = valid_candidates[idx]["id"]
                visual_scores[cand_id] = float(score)

        logger.info(f"Visual similarity evaluation completed for {len(visual_scores)} candidate products.")
    except Exception as e:
        logger.warning(f"Visual similarity evaluation error: {e}. Using family matching scores.")

    return visual_scores


def match_catalog_products(ident_result, catalog, user_image_bytes=None, user_mime_type="image/jpeg"):
    """
    Pipeline:
    1. Shortlist all site products belonging to the same plant family/genus.
    2. Visually crosscheck user image against shortlisted product images.
    3. Return closest 8 matches with scores.
    """
    common_names = ident_result.get("common_names", [])
    scientific_name = ident_result.get("scientific_name", "")
    genus = ident_result.get("genus", "")
    family_name = ident_result.get("family_name", "")
    visual_traits = ident_result.get("visual_traits", "")

    # Step 1: Shortlist all products of the same family
    shortlisted = shortlist_family_products(ident_result, catalog)
    candidate_products = [prod for prod, family_score in shortlisted]

    # Step 2: Crosscheck user given image with shortlisted product images
    visual_scores = {}
    if user_image_bytes and candidate_products:
        visual_scores = evaluate_visual_similarity(user_image_bytes, user_mime_type, candidate_products)

    # Step 3: Combine scores (75% visual crosscheck, 25% family match relevance)
    final_scored_products = []
    for prod, family_score in shortlisted:
        prod_id = prod.get("id")
        if prod_id in visual_scores:
            vis_score = visual_scores[prod_id]
            combined_score = (0.75 * vis_score) + (0.25 * family_score)
        else:
            combined_score = family_score
        final_scored_products.append((prod, combined_score))

    final_scored_products.sort(key=lambda x: x[1], reverse=True)

    # Step 4: Closest 8 matches
    top_matches = final_scored_products[:8]

    # Always mark confident if catalog matches exist so UI displays products grid
    is_confident = len(top_matches) > 0

    # Format result items
    formatted_matches = []
    for prod, score in top_matches:
        images = prod.get("images", [])
        img_url = ""
        if images:
            first_img = images[0].get("src", "")
            img_url = first_img + ("&width=400" if "?" in first_img else "?width=400") if first_img else ""
        
        variants = prod.get("variants", [])
        price = variants[0].get("price", "0.00") if variants else "0.00"

        handle = prod.get("handle", "")
        prod_url = f"https://greenxonline.com/products/{handle}" if handle else "#"

        formatted_matches.append({
            "id": prod.get("id"),
            "title": prod.get("title"),
            "url": prod_url,
            "image": img_url,
            "price": price,
            "match_score": int(round(max(score, 30.0)))
        })

    identified_as = scientific_name if scientific_name else (common_names[0] if common_names else "Plant")

    return {
        "identified_as": identified_as,
        "genus": genus,
        "family_name": family_name,
        "visual_traits": visual_traits,
        "confident": is_confident,
        "family_shortlisted_count": len(shortlisted),
        "matches": formatted_matches
    }


@app.route("/health", methods=["GET"])
@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint for Railway deployment monitoring."""
    return jsonify({
        "status": "ok",
        "service": "plant-lens-api",
        "catalog_cached_items": len(_catalog_cache["products"])
    }), 200


@app.route("/api/plant-lens/identify", methods=["POST"])
@limiter.limit("20 per hour")
def identify_plant():
    """
    Public customer-facing endpoint to identify plant from uploaded photo and return catalog matches.
    """
    if "photo" not in request.files:
        return jsonify({"error": "Missing 'photo' file field in multipart request"}), 400

    file = request.files["photo"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Content length / file size verification
    file_bytes = file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return jsonify({"error": "File size exceeds 5MB limit"}), 400

    if len(file_bytes) == 0:
        return jsonify({"error": "Uploaded file is empty"}), 400

    # Content type verification
    mimetype = file.mimetype or "image/jpeg"
    ext = os.path.splitext(file.filename)[1].lower()
    valid_exts = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".bmp"]
    
    if not mimetype.startswith("image/") and ext not in valid_exts:
        return jsonify({"error": "Invalid file type. Please upload a valid image file."}), 400

    # Step 1 — Identify with Gemini Vision
    try:
        logger.info(f"Received plant image: {file.filename} ({len(file_bytes)} bytes, {mimetype})")
        ident_result = call_gemini_vision(file_bytes, mimetype)
        logger.info(f"Gemini identification: {ident_result}")
    except Exception as e:
        logger.error(f"Gemini identification error: {e}")
        return jsonify({"error": "Couldn't identify that photo — try a clearer shot"}), 500

    # Step 2 & 3 — Match against Shopify catalog & perform visual similarity check
    try:
        catalog = get_shopify_catalog()
        response_payload = match_catalog_products(ident_result, catalog, user_image_bytes=file_bytes, user_mime_type=mimetype)
        return jsonify(response_payload), 200
    except Exception as e:
        logger.error(f"Catalog matching error: {e}")
        return jsonify({"error": "Error matching plant identification with product catalog"}), 500


@app.route("/api/plant-lens/refresh-catalog", methods=["POST"])
def refresh_catalog():
    """
    Internal endpoint to force-refresh the in-memory Shopify catalog cache. Gated by master key.
    """
    provided_key = request.headers.get("X-Fawa-Key") or request.args.get("fawa") or request.form.get("fawa")
    if provided_key != FAWA_MASTER_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    products = get_shopify_catalog(force_refresh=True)
    return jsonify({
        "status": "success",
        "message": "Shopify product catalog refreshed successfully",
        "product_count": len(products)
    }), 200


@app.errorhandler(429)
def ratelimit_handler(e):
    """Custom error response for IP rate limit exceeding."""
    return jsonify({
        "error": "Rate limit exceeded (max 20 searches/hour per IP). Please try again later."
    }), 429


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
