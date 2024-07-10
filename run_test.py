import unittest

# Discover all tests in the 'tests' directory matching the pattern 'test_*.py'
loader = unittest.TestLoader()
suite = loader.discover("tests", pattern="test_*.py")

# Run the test suite
runner = unittest.TextTestRunner()
runner.run(suite)
