import unittest
from unittest.mock import patch
import json
import os
import io
from app import app, clean_json_response, match_catalog_products, _catalog_cache

class PlantLensApiTestCase(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        # Seed mock catalog cache
        _catalog_cache['products'] = [
            {
                "id": 101,
                "title": "Monstera Deliciosa - Swiss Cheese Plant",
                "handle": "monstera-deliciosa",
                "tags": ["houseplant", "tropical", "monstera"],
                "product_type": "Indoor Plant",
                "images": [{"src": "https://example.com/monstera.jpg"}],
                "variants": [{"price": "45.00"}]
            },
            {
                "id": 102,
                "title": "Snake Plant Laurentii - Sansevieria Trifasciata",
                "handle": "snake-plant-laurentii",
                "tags": ["snake plant", "sansevieria", "succulent"],
                "product_type": "Indoor Plant",
                "images": [{"src": "https://example.com/snake.jpg"}],
                "variants": [{"price": "29.99"}]
            },
            {
                "id": 103,
                "title": "Fiddle Leaf Fig - Ficus Lyrata",
                "handle": "fiddle-leaf-fig",
                "tags": ["fig", "ficus lyrata", "tree"],
                "product_type": "Indoor Plant",
                "images": [{"src": "https://example.com/fiddle.jpg"}],
                "variants": [{"price": "65.00"}]
            }
        ]
        _catalog_cache['timestamp'] = 9999999999

    def test_health_check(self):
        """Test health check route."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'ok')
        self.assertEqual(data.get('service'), 'plant-lens-api')

    def test_clean_json_response(self):
        """Test stripping markdown code block fences from LLM responses."""
        raw_json_fenced = "```json\n{\n  \"common_names\": [\"Snake Plant\"]\n}\n```"
        cleaned = clean_json_response(raw_json_fenced)
        self.assertTrue(cleaned.startswith('{'))
        self.assertTrue(cleaned.endswith('}'))
        parsed = json.loads(cleaned)
        self.assertEqual(parsed['common_names'], ["Snake Plant"])

    def test_missing_file_upload(self):
        """Test 400 when photo field is missing."""
        response = self.client.post('/api/plant-lens/identify', data={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('Missing', data.get('error', ''))

    def test_over_5mb_file_upload(self):
        """Test 400 when uploaded file exceeds 5MB limit."""
        large_content = b"0" * (5 * 1024 * 1024 + 10)
        data = {
            'photo': (io.BytesIO(large_content), 'large_image.jpg', 'image/jpeg')
        }
        response = self.client.post('/api/plant-lens/identify', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        res_data = json.loads(response.data)
        self.assertIn('5MB limit', res_data.get('error', ''))

    def test_invalid_file_type_upload(self):
        """Test 400 when non-image file is uploaded."""
        data = {
            'photo': (io.BytesIO(b"hello world text"), 'file.txt', 'text/plain')
        }
        response = self.client.post('/api/plant-lens/identify', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        res_data = json.loads(response.data)
        self.assertIn('valid image', res_data.get('error', ''))

    def test_catalog_matching_high_confidence(self):
        """Test catalog family matching with exact/high species match."""
        ident = {
            "common_names": ["Swiss Cheese Plant", "Monstera"],
            "scientific_name": "Monstera deliciosa",
            "genus": "Monstera",
            "family_name": "Araceae",
            "family_keywords": ["Monstera", "Araceae"],
            "confidence": "high"
        }
        result = match_catalog_products(ident, _catalog_cache['products'])
        self.assertEqual(result['identified_as'], "Monstera deliciosa")
        self.assertTrue(result['confident'])
        self.assertGreater(len(result['matches']), 0)
        self.assertEqual(result['matches'][0]['id'], 101)
        self.assertEqual(result['matches'][0]['title'], "Monstera Deliciosa - Swiss Cheese Plant")

    def test_family_shortlisting_and_visual_matches(self):
        """Test that site products are shortlisted by family name and matches capped at 8."""
        ident = {
            "common_names": ["Crystal Anthurium"],
            "scientific_name": "Anthurium crystallinum",
            "genus": "Anthurium",
            "family_name": "Araceae",
            "family_keywords": ["Anthurium", "Araceae"],
            "confidence": "high"
        }
        result = match_catalog_products(ident, _catalog_cache['products'])
        self.assertEqual(result['family_name'], "Araceae")
        self.assertGreaterEqual(result['family_shortlisted_count'], 1)
        self.assertLessEqual(len(result['matches']), 8)

    def test_catalog_matching_low_confidence_fallback(self):
        """Test catalog matching with unknown/unmatched plant."""
        ident = {
            "common_names": ["Unusual Desert Moss"],
            "scientific_name": "Unknown species",
            "confidence": "low"
        }
        result = match_catalog_products(ident, _catalog_cache['products'])
        self.assertFalse(result['confident'])
        self.assertGreater(len(result['matches']), 0)

    def test_refresh_catalog_unauthorized(self):
        """Test 401 on refresh catalog endpoint without valid master key."""
        response = self.client.post('/api/plant-lens/refresh-catalog')
        self.assertEqual(response.status_code, 401)

    def test_refresh_catalog_authorized(self):
        """Test refresh catalog endpoint with valid master key."""
        with patch('app.get_shopify_catalog') as mock_fetch:
            mock_fetch.return_value = _catalog_cache['products']
            response = self.client.post('/api/plant-lens/refresh-catalog', headers={'X-Fawa-Key': 'fawa'})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data.get('status'), 'success')

if __name__ == '__main__':
    unittest.main()
