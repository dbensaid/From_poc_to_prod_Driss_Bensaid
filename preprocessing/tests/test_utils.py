import unittest
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # TODO: CODE HERE
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        # Calculus made 100*0.8 / 20=4
        self.assertEqual(base._get_num_train_batches(), 4)

    def test__get_num_test_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # TODO: CODE HERE
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        # Calculus made (100 - 80) / 20 = 1
        self.assertEqual(base._get_num_test_batches(), 1)


    def test_get_index_to_label_map(self):
        # TODO: CODE HERE
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return a list [value1,value2]
        base._get_label_list = MagicMock(return_value=['value_1', 'value_2'])
        print(base._get_label_list())
        x = dict({
            0: 'value_1',
            1: 'value_2',
        })
        print(x)
        self.assertEqual(base.get_index_to_label_map(), x)



    def test_index_to_label_and_label_to_index_are_identity(self):
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return a list [value1,value2]
        base._get_label_list = MagicMock(return_value= ['value_1', 'value_2'])
        x = dict({
            'value_1':0,
            'value_2':1,
        })
        self.assertEqual(base.get_label_to_index_map(), x)


    def test_to_indexes(self):
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return a list [value1,value2]
        base._get_label_list = MagicMock(return_value=['value_1', 'value_2','value_3'])
        base.get_index_to_label_map()
        x = [0, 2]
        self.assertEqual(base.to_indexes(['value_1','value_3']), x)


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_a'],
            'tag_id': [1, 2],
            'tag_position': [0, 0],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset("fake_path", batch_size=1, train_ratio=0.5, min_samples_per_label=1)
        expected= int(2)
        print(dataset._get_num_samples())
        self.assertEqual(dataset._get_num_samples(), expected)

        #self.assertEqual(dataset.get_num_samples(), x)

    def test_get_train_batch_returns_expected_shape(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2','id_3','id_4'],
            'tag_name': ['tag_a', 'tag_a', 'tag_b', 'tag_b'],
            'tag_id': [1, 1, 2,2],
            'tag_position': [0, 0, 0, 0],
            'title': ['title_1', 'title_2','title_3','title_4']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset("fake_path", batch_size=2, train_ratio=0.5, min_samples_per_label=1)
        x_batch, y_batch = dataset.get_train_batch()
        print(x_batch)
        print(y_batch.shape)
        # Assert that the shape of the returned data is as expected
        self.assertEqual(len(x_batch), 2)
        self.assertEqual(y_batch.shape, (2, 2))



    def test_get_test_batch_returns_expected_shape(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2','id_3','id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_a', 'tag_b', 'tag_b','tag_c', 'tag_c'],
            'tag_id': [1, 1, 2,2,3,3],
            'tag_position': [0, 0, 0, 0,0,0],
            'title': ['title_1', 'title_2','title_3','title_4','title_5','title_6']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset("fake_path", batch_size=1, train_ratio=0.5,
                                                       min_samples_per_label=1)

        # Get a batch of data from the mock dataset
        x_batch, y_batch = dataset.get_test_batch()
        print(x_batch)
        print(y_batch)
        # Assert that the shape of the returned data is as expected
        self.assertEqual(len(x_batch), 1)
        self.assertEqual(y_batch.shape, (1, 3))

    def test_get_train_batch_raises_assertion_error(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_a'],
            'tag_id': [1, 2],
            'tag_position': [0, 0],
            'title': ['title_1', 'title_2']
        }))
        # Assert an impossible scenario (batch > the dataset)
        with self.assertRaises(AssertionError):
            utils.LocalTextCategorizationDataset("fake_path", batch_size=3, train_ratio=0.5, min_samples_per_label=1)

