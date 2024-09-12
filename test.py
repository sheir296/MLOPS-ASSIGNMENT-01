
import unittest
from main import preprocess_data, load_data, predict

class TestModel(unittest.TestCase):
    def test_preprocess_data(self):
        data = load_data('data/house_prices.csv')
        X, y = preprocess_data(data)
        self.assertEqual(X.shape[1], 3)
        self.assertTrue('price' in data.columns)
    
    def test_prediction(self):
        result = predict(1200, 3, 2)  # Example input
        self.assertIsInstance(result, float)

if __name__ == '__main__':
    unittest.main()
