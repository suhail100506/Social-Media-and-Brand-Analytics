"""
Unit tests for all Flask API endpoints using test_client.
Verifies all 10 required routes, input validation, query parameters, and JSON payloads.
"""
import unittest
import json
from app import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_health(self):
        res = self.client.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('data_loaded'))
        self.assertTrue(data.get('models_loaded'))
        self.assertEqual(data.get('total_posts'), 2216)
        print("[OK] /api/health passed")

    def test_summary(self):
        res = self.client.get('/api/summary')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['total_posts'], 2216)
        self.assertEqual(data['total_engagement'], 101707.0)
        self.assertIn('most_discussed_topic', data)
        self.assertIn('highest_engagement_topic', data)
        self.assertIn('most_negative_topic', data)
        self.assertIn('most_positive_topic', data)
        self.assertIn('most_active_platform', data)
        self.assertEqual(data['spike_count'], 200)

        # Test with filter
        filtered_res = self.client.get('/api/summary?topic=Customer Service')
        self.assertEqual(filtered_res.status_code, 200)
        f_data = filtered_res.get_json()
        self.assertEqual(f_data['total_posts'], 446)
        print("[PASS] /api/summary passed (unfiltered & filtered)")

    def test_topics(self):
        res = self.client.get('/api/topics')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 5)
        topics = [t['topic'] for t in data]
        self.assertIn('Customer Service', topics)
        self.assertIn('Product', topics)
        print("[PASS] /api/topics passed")

    def test_sentiment(self):
        res = self.client.get('/api/sentiment')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('overall', data)
        self.assertIn('topic_vs_sentiment', data)
        self.assertIn('platform_vs_sentiment', data)
        self.assertEqual(data['overall']['positive'], 883)
        self.assertEqual(data['overall']['neutral'], 662)
        self.assertEqual(data['overall']['negative'], 638)
        print("[PASS] /api/sentiment passed")

    def test_engagement(self):
        res = self.client.get('/api/engagement')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('by_topic', data)
        self.assertIn('by_platform', data)
        self.assertEqual(len(data['by_topic']), 5)
        print("[PASS] /api/engagement passed")

    def test_trends(self):
        # Daily
        res = self.client.get('/api/trends?freq=D')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 500)
        # Weekly
        res_w = self.client.get('/api/trends?freq=W')
        self.assertEqual(res_w.status_code, 200)
        data_w = res_w.get_json()
        self.assertIsInstance(data_w, list)
        self.assertGreater(len(data_w), 90)
        print("[PASS] /api/trends passed (Daily & Weekly)")

    def test_spikes(self):
        res = self.client.get('/api/spikes')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 200)
        # Verify sorted desc
        self.assertGreaterEqual(data[0]['spike_ratio'], data[-1]['spike_ratio'])

        # With min_ratio
        res_filter = self.client.get('/api/spikes?min_ratio=2.5')
        self.assertEqual(res_filter.status_code, 200)
        data_filter = res_filter.get_json()
        for item in data_filter:
            self.assertGreaterEqual(item['spike_ratio'], 2.5)
        print("[PASS] /api/spikes passed")

    def test_posts(self):
        res = self.client.get('/api/posts?page=1&limit=10')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data['posts']), 10)
        self.assertEqual(data['total'], 2216)
        self.assertEqual(data['page'], 1)
        self.assertEqual(data['limit'], 10)

        # Search query
        res_q = self.client.get('/api/posts?q=frustrating')
        self.assertEqual(res_q.status_code, 200)
        data_q = res_q.get_json()
        self.assertGreater(data_q['total'], 0)
        print("[PASS] /api/posts passed (pagination & search)")

    def test_model_metrics(self):
        res = self.client.get('/api/model-metrics')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('accuracy', data)
        self.assertIn('precision', data)
        self.assertIn('recall', data)
        print("[PASS] /api/model-metrics passed")

    def test_predict(self):
        # 1. Positive post
        payload = {"text": "Absolutely loving the new platform update, this is amazing!"}
        res = self.client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['sentiment'], 'positive')
        self.assertGreater(data['confidence'], 80)
        self.assertIn('positive', data['probabilities'])
        self.assertIn('negative', data['probabilities'])
        self.assertIn('neutral', data['probabilities'])

        # 2. Negative post
        payload_neg = {"text": "Worst brand experience I've had in a long time, frustrating."}
        res_neg = self.client.post('/api/predict', data=json.dumps(payload_neg), content_type='application/json')
        self.assertEqual(res_neg.status_code, 200)
        data_neg = res_neg.get_json()
        self.assertEqual(data_neg['sentiment'], 'negative')
        self.assertGreater(data_neg['confidence'], 80)

        # 3. Validation - Empty text
        res_empty = self.client.post('/api/predict', data=json.dumps({"text": ""}), content_type='application/json')
        self.assertEqual(res_empty.status_code, 400)

        # 4. Validation - Too long text
        res_long = self.client.post('/api/predict', data=json.dumps({"text": "a" * 1001}), content_type='application/json')
        self.assertEqual(res_long.status_code, 400)
        print("[PASS] /api/predict passed (positive, negative, and validation)")

if __name__ == '__main__':
    unittest.main()
