import unittest

# Discover all tests in the 'tests' directory matching the pattern 'test_*.py'
loader = unittest.TestLoader()
suite = loader.discover("lib/tests", pattern="test_*.py")

# Run the test suite
runner = unittest.TextTestRunner()
output = runner.run(suite)

# if any errors or failures then exit failure
if output.errors or output.failures:
    exit(1)
