import argparse
import unittest

from config.test.test_config import test_config


def main():
    parser = argparse.ArgumentParser(description="Toggle the online flag.")
    parser.add_argument("--online", action="store_true", help="Toggle the online flag")

    args = parser.parse_args()

    test_config.ONLINE = args.online

    # Discover all tests in the 'tests' directory matching the pattern 'test_*.py'
    loader = unittest.TestLoader()
    # re = "(.+_test\.py)"
    # if test_config.ONLINE:
    #     re += "|(test_.+\.py)"
    suite = loader.discover("lib/tests", pattern="test_*.py")

    # Run the test suite
    runner = unittest.TextTestRunner()
    output = runner.run(suite)
    # if any errors or failures then exit failure
    if output.errors or output.failures:
        exit(1)

    if not test_config.ONLINE:
        suite = loader.discover("lib/tests", pattern="*_test.py")
        runner = unittest.TextTestRunner()
        output = runner.run(suite)
        if output.errors or output.failures:
            exit(1)


if __name__ == "__main__":
    main()
