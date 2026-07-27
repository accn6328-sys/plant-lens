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

# Global in-memory catalog cache
_catalog_cache = {
    "timestamp": 0,
    "products": []
}


def get_shopify_catalog(force_refresh=False):
    """
    Fetch the full product catalog from Shopify store (/products.json) with pagination.
    Caches results in-memory with TTL.
    """
    global _catalog_cache
    now = time.time()
    
    if not force_refresh and _catalog_cache["products"] and (now - _catalog_cache["timestamp"] < CATALOG_TTL_SECONDS):
        logger.info(f"Returning cached catalog with {len(_catalog_cache['products'])} products.")
        return _catalog_cache["products"]

    logger.info(f"Fetching fresh product catalog from {SHOPIFY_CATALOG_URL}...")
    products = []
    page = 1
    
    try:
        while True:
            separator = "&" if "?" in SHOPIFY_CATALOG_URL else "?"
            url = f"{SHOPIFY_CATALOG_URL}{separator}limit=250&page={page}"
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                logger.error(f"Shopify catalog fetch failed on page {page} with status {resp.status_code}")
                break
            
            data = resp.json()
            page_products = data.get("products", [])
            if not page_products:
                break
                
            products.extend(page_products)
            logger.info(f"Page {page}: fetched {len(page_products)} products (Total: {len(products)})")
            
            if len(page_products) < 250:
                break
            page += 1
            
        if products:
            _catalog_cache["timestamp"] = now
            _catalog_cache["products"] = products
            logger.info(f"Successfully cached {len(products)} catalog products.")
        elif _catalog_cache["products"]:
            logger.warning("Fresh catalog fetch yielded 0 items; retaining previous cached catalog.")
            return _catalog_cache["products"]

    except Exception as e:
        logger.error(f"Error fetching catalog from Shopify: {e}")
        if _catalog_cache["products"]:
            logger.info("Serving stale catalog cache due to fetch error.")
            return _catalog_cache["products"]

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
    family name, visual traits, and AT MOST 12 related species of the same family/genus.
    Supports both google-genai and google-generativeai SDKs.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    prompt = (
        "You are identifying a houseplant from a customer photo for a plant store's visual search feature.\n"
        "Look at the plant in this image and respond with ONLY valid JSON, no markdown fences, no preamble, in this exact JSON format:\n\n"
        "{\n"
        '  "common_names": ["most likely common name", "alternate common name"],\n'
        '  "scientific_name": "Genus species",\n'
        '  "family_name": "Family name (e.g. Araceae)",\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "visual_traits": "one short sentence describing leaf shape, color pattern, growth habit",\n'
        '  "related_species": [\n'
        '    {\n'
        '      "species": "Genus species2",\n'
        '      "relationship_features": "Short description of relationship and notable features"\n'
        '    }\n'
        '  ]\n'
        "}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. List 1-3 common_names ordered by likelihood.\n"
        "2. Include AT MOST 12 related species of the SAME family or genus as the identified plant (e.g., if identified as Anthurium crystallinum, list up to 12 related Anthurium species like Anthurium magnificum, Anthurium regale, Anthurium clarinervium, etc.).\n"
        "3. For each related species, provide its scientific name in 'species' and visual/taxonomic relationship in 'relationship_features'.\n"
        "4. If you cannot identify the plant with high confidence, set confidence to 'low' and still give your best guess and related species — never refuse to answer."
    )

    raw_text = ""
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    # Attempt using new google.genai SDK
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
    except Exception as err_new_sdk:
        logger.warning(f"google.genai SDK call failed/unavailable ({err_new_sdk}), falling back to google.generativeai...")
        
        # Fallback to google.generativeai SDK
        try:
            import google.generativeai as genai_old
            genai_old.configure(api_key=api_key)
            model = genai_old.GenerativeModel(model_name)
            
            image_part = {
                "mime_type": mime_type,
                "data": image_bytes
            }
            response = model.generate_content([image_part, prompt])
            raw_text = response.text
        except Exception as err_old_sdk:
            logger.error(f"Both Gemini SDKs failed: {err_old_sdk}")
            raise err_old_sdk

    cleaned = clean_json_response(raw_text)
    try:
        data = json.loads(cleaned)
        if "related_species" in data and isinstance(data["related_species"], list):
            data["related_species"] = data["related_species"][:12]
        else:
            data["related_species"] = []
        return data
    except json.JSONDecodeError as jde:
        logger.error(f"Failed to parse Gemini response as JSON: {cleaned}")
        raise ValueError(f"Invalid JSON returned by Gemini Vision: {jde}")


def evaluate_visual_similarity(user_image_bytes, user_mime_type, candidate_products):
    """
    Downloads candidate product images and calls Gemini Vision to score visual similarity (0-100)
    between the user's uploaded plant image and candidate product images based on foliage visual traits.
    Returns dict mapping product_id -> visual_score (0-100).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not user_image_bytes:
        return {}

    visual_scores = {}
    valid_candidates = []

    # Download images for candidate products (up to 10 candidates for speed)
    for prod in candidate_products[:10]:
        prod_id = prod.get("id")
        images = prod.get("images", [])
        if not images:
            continue
        img_url = images[0].get("src")
        if not img_url:
            continue
        
        try:
            resp = requests.get(img_url, timeout=2.5)
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
        "The following images are candidate product photos from our catalog.\n"
        "Compare each candidate product image against Image 0 and rate its visual similarity from 0 to 100 "
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
        logger.warning(f"Visual similarity evaluation error: {e}. Using text matching scores.")

    return visual_scores


def match_catalog_products(ident_result, catalog, user_image_bytes=None, user_mime_type="image/jpeg"):
    """
    Fuzzy-matches identified plant & up to 12 related family species against product catalog text,
    visually compares matching product images against user image, and returns closest 8 product matches.
    """
    common_names = ident_result.get("common_names", [])
    scientific_name = ident_result.get("scientific_name", "")
    family_name = ident_result.get("family_name", "")
    visual_traits = ident_result.get("visual_traits", "")
    confidence_level = ident_result.get("confidence", "low").lower()
    related_species = ident_result.get("related_species", [])

    if not isinstance(related_species, list):
        related_species = []

    # Clean related_species to ensure max 12 items
    clean_related_species = []
    for rel in related_species[:12]:
        if isinstance(rel, dict):
            sp_name = rel.get("species", rel.get("name", ""))
            rel_feat = rel.get("relationship_features", rel.get("notable_features", rel.get("description", "")))
            if sp_name:
                clean_related_species.append({
                    "species": sp_name,
                    "relationship_features": rel_feat
                })

    candidates = []
    if scientific_name:
        candidates.append(scientific_name)
    for name in common_names:
        if name and name not in candidates:
            candidates.append(name)
    for rel in clean_related_species:
        sp = rel.get("species", "")
        if sp and sp not in candidates:
            candidates.append(sp)

    if not candidates:
        candidates = ["plant"]

    scored_products = []
    
    for product in catalog:
        title = product.get("title", "")
        tags = product.get("tags", [])
        if isinstance(tags, list):
            tags_str = " ".join(tags)
        else:
            tags_str = str(tags)
        product_type = product.get("product_type", "")
        
        # Combine text fields into searchable string
        search_blob = f"{title} {tags_str} {product_type}".lower()

        best_candidate_score = 0.0
        for candidate in candidates:
            cand_lower = candidate.lower()
            # Calculate match ratio using rapidfuzz token_set_ratio & partial_ratio
            token_score = fuzz.token_set_ratio(cand_lower, search_blob)
            partial_score = fuzz.partial_ratio(cand_lower, search_blob)
            cand_score = max(token_score, partial_score)

            # Bonus points if candidate word appears in title
            if cand_lower in title.lower():
                cand_score = min(100.0, cand_score + 15)

            if cand_score > best_candidate_score:
                best_candidate_score = cand_score

        scored_products.append((product, best_candidate_score))

    # Sort descending by match score
    scored_products.sort(key=lambda x: x[1], reverse=True)
    
    # Filter candidate products for visual matching
    candidate_products = [p for p, score in scored_products if score >= 30][:15]
    if not candidate_products and scored_products:
        candidate_products = [p for p, score in scored_products[:8]]

    # Perform visual image similarity comparison if user image bytes provided
    visual_scores = {}
    if user_image_bytes and candidate_products:
        visual_scores = evaluate_visual_similarity(user_image_bytes, user_mime_type, candidate_products)

    # Combine text score and visual similarity score
    final_scored_products = []
    for product, text_score in scored_products:
        prod_id = product.get("id")
        if prod_id in visual_scores:
            vis_score = visual_scores[prod_id]
            combined_score = (0.7 * vis_score) + (0.3 * text_score)
        else:
            combined_score = text_score
        final_scored_products.append((product, combined_score))

    final_scored_products.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 8 closest matches
    top_matches = final_scored_products[:8]

    top_score = top_matches[0][1] if top_matches else 0
    is_confident = (top_score >= 50) and (confidence_level != "low")

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
            "match_score": int(round(score))
        })

    # Pick best display name
    identified_as = scientific_name if scientific_name else (common_names[0] if common_names else "Plant")

    return {
        "identified_as": identified_as,
        "family_name": family_name,
        "visual_traits": visual_traits,
        "confident": is_confident,
        "related_species": clean_related_species[:12],
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
