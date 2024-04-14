import unittest
import argparse

parser = argparse.ArgumentParser(description='run unit tests for MP.')
parser.add_argument('-j', '--json', action='store_true', help='Results in Gradescope JSON format.')
args = parser.parse_args()

suite = unittest.defaultTestLoader.discover('tests', pattern='test_visible_extra.py')

if args.json:
    from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner
    JSONTestRunner(visibility='visible').run(suite)
else:
    result = unittest.TextTestRunner().run(suite)