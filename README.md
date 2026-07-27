# Plant Lens API & Shopify Integration

Plant Lens visual search service powered by Google Gemini 2.5 Flash Lite Vision and RapidFuzz product catalog matching.

## Architecture

1. **Identify** (`POST /api/plant-lens/identify`): Image uploaded by customer is passed to Gemini Vision model (`gemini-2.5-flash-lite`), which identifies common names, scientific name, visual traits, and confidence.
2. **Match**: Identification names are fuzzy-matched (`rapidfuzz`) against the Shopify product catalog (`title`, `tags`, `product_type`).
3. **Response**: Ranked top 6 matches returned to frontend JS with `match_score` and confidence flag.

## API Endpoints

- `GET /health` or `GET /` — Service health check and cache status.
- `POST /api/plant-lens/identify` — Public endpoint accepting `multipart/form-data` with `photo` field.
  - Rate limit: 20 reqs/hr per IP.
  - File size: max 5MB.
- `POST /api/plant-lens/refresh-catalog` — Internal endpoint to force refresh the Shopify catalog cache.
  - Headers: `X-Fawa-Key: fawa` (or query param `?fawa=fawa`).

## Environment Variables (Railway)

- `GEMINI_API_KEY`: Gemini API key (`AQ.Ab8RN6KKRHrWInGt6DxWJPuPox-X1ujvuuonztlMTT-GQgIYsg`).
- `GEMINI_MODEL`: `gemini-2.5-flash-lite`.
- `ALLOWED_ORIGIN`: `https://greenxonline.com` (CORS restriction).
- `SHOPIFY_CATALOG_URL`: `https://greenxonline.com/products.json`.
- `FAWA_MASTER_KEY`: `fawa`.

## Local Development & Testing

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run test suite:
   ```bash
   python test_app.py
   ```
3. Start development server:
   ```bash
   python app.py
   ```

## Railway Deployment

1. Create a new service on Railway connected to this repository/directory (`plant-lens-api`).
2. Add environment variables: `GEMINI_API_KEY`, `GEMINI_MODEL`, `ALLOWED_ORIGIN`, `SHOPIFY_CATALOG_URL`, `FAWA_MASTER_KEY`.
3. Deployment uses `Procfile` (`web: gunicorn app:app`) and `railway.json`.

## Shopify Theme Files

- `assets/plant-lens.js`: Clean JS modal handler calling Railway backend URL (`CONFIG.apiBaseUrl`).
- `snippets/plant-lens-modal.liquid`: Liquid snippet modal markup (MobileNet and TensorFlow script tags removed).
