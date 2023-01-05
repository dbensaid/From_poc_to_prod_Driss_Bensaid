import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from preprocessing.preprocessing import utils
from train.train import run
from predict.predict.run import TextPredictionModel

def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })

class TestTextPredictionModel(unittest.TestCase):


    #dataset = utils.LocalTextCategorizationDataset.load_dataset

    def test_predict(self):
        # TODO: CODE HERE
        # create a dictionary params for train config
        params = {
            "batch_size": 2,
            "epochs": 1,
            "dense_dim": 64,
            "min_samples_per_label": 1,
            "verbose": 1
        }
        # use the function defined on test_model_train as a mock for utils.LocalTextCategorizationDataset.load_dataset
        utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, path_artefact = run.train('fake_path', params, model_path=r"\train\data\artefacts", add_timestamp=True)
            # instance a TextPredictModel class
            prediction_model = TextPredictionModel.from_artefacts(path_artefact)

            # run a prediction
            prediction = prediction_model.predict(['This is totally a post about php'], top_k=1)

        # assert that predictions obtained are equals to expected ones
        self.assertEqual(['php'], prediction)
        #for the prediction top_k=1
