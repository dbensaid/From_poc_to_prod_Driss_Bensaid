import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from predict.predict.run import TextPredictionModel

class TestTextPredictionModel(unittest.TestCase):
    def setUp(self):
        # Load training artifacts and instantiate the TextPredictionModel object
        self.model = TextPredictionModel.from_artefacts('train/tests/fake_path/2023-01-04-18-05-19')

    def test_predict(self):
        # Test the predict method with different input texts and expected output
        self.assertEqual(self.model.predict(['This is a test text']), [{'tag1': 0.9, 'tag2': 0.1}])
        self.assertEqual(self.model.predict(['Another test text']), [{'tag2': 0.7, 'tag3': 0.3}])