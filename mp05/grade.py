import unittest, argparse
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner


parser = argparse.ArgumentParser(description="CS440/ECE448 MP: Neural Nets and PyTorch")
parser.add_argument("--json", action="store_true", help="Gradescope JSON format.")


def main():
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover("tests")
    if args.json:
        JSONTestRunner(visibility="visible").run(suite)
    else:
        unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()
