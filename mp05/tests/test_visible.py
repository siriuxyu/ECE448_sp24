from gradescope_utils.autograder_utils.decorators import partial_credit, weight
import numpy as np
import submitted
import torch.nn as nn
import traceback
import unittest

import helper


class Test(unittest.TestCase):
    def setUp(self):
        helper.init_seeds(42)
        train_set, train_labels, test_set, test_labels = helper.Load_dataset(
            "data/mp_data"
        )
        train_set, test_set = helper.Preprocess(train_set, test_set)
        train_loader, test_loader = helper.Get_DataLoaders(
            train_set, train_labels, test_set, test_labels, 100
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_set = test_set
        self.test_labels = test_labels

    @weight(20)
    def test_sequential_layers(self):
        try:
            student_layers: nn.Sequential = submitted.create_sequential_layers()
            self.assertIsInstance(
                student_layers, nn.Sequential, "Not an nn.Sequential object"
            )
            self.assertEqual(len(student_layers), 3, "Incorrect number of layers")
            self.assertIsInstance(student_layers[0], nn.Linear, "Layer 0 is not linear")
            self.assertEqual(
                student_layers[0].in_features, 2, "Incorrect input features in layer 0"
            )
            self.assertEqual(
                student_layers[0].out_features,
                3,
                "Incorrect output features in layer 0",
            )
            self.assertIsInstance(
                student_layers[1], nn.Sigmoid, "Layer 1 is not a Sigmoid"
            )
            self.assertIsInstance(student_layers[2], nn.Linear, "Layer 2 is not linear")
            self.assertEqual(
                student_layers[2].in_features, 3, "Incorrect input features in layer 2"
            )
            self.assertEqual(
                student_layers[2].out_features,
                5,
                "Incorrect output features in layer 2",
            )
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.assertTrue(False, "Error in sequential layers. Run locally to debug.")

    @weight(20)
    def test_loss_fn(self):
        try:
            loss_fn = submitted.create_loss_function()
            self.assertIsInstance(loss_fn, nn.modules.loss._Loss)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.assertTrue(False, "Error in loss function. Run locally to debug.")

    @partial_credit(30)
    def test_accuracy(self, set_score=None):
        # Train
        try:
            model = submitted.train(self.train_loader, 50)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.assertTrue(
                False, "Error in neural net implementation. Run locally to debug."
            )
        # Predict
        pred_values = model(self.test_set)  # Predicted value of the testing set
        pred_values = pred_values.detach().numpy()
        pred_labels = np.argmax(pred_values, axis=1)  # Predicted labels
        # Error handling
        self.assertEquals(
            len(pred_labels),
            len(self.test_labels),
            "Incorrect size of predicted labels.",
        )
        num_parameters = sum([np.prod(w.shape) for w in model.parameters()])
        upper_threshold = 10000
        lower_threshold = 1000000
        print("Total number of network parameters: ", num_parameters)
        self.assertLess(
            num_parameters,
            lower_threshold,
            "Your network is way too large with "
            + str(num_parameters)
            + " parameters. The upper limit is "
            + str(lower_threshold)
            + "!",
        )
        self.assertGreater(
            num_parameters,
            upper_threshold,
            "Your network is suspiciously compact. Have you implemented something other than a neural network?"
            + " Or perhaps the number of hidden neurons is too small. Neural nets usually have over "
            + str(upper_threshold)
            + " parameters!",
        )
        # Accuracy test
        accuracy, conf_mat = helper.compute_accuracies(pred_labels, self.test_labels)
        print("\n Accuracy:", accuracy)
        print("\nConfusion Matrix = \n {}".format(conf_mat))
        # Compute score
        score = 0
        for threshold in [0.15, 0.25, 0.48, 0.55, 0.57, 0.61]:
            if accuracy >= threshold:
                score += 5
                print("+5 points for accuracy above", str(threshold))
            else:
                break
        if score != 30:
            print("Accuracy must be above 0.61")
        set_score(score)
