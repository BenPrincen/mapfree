import unittest
import os
from lib.eval.sift_runner import SiftRunner


class TestSiftRunner(unittest.TestCase):
    def test_creation(self):
        config_path = os.path.join("config/sift", "sift_config.yaml")
        sift_runner = SiftRunner(config_path)
        self.assertTrue(isinstance(sift_runner, SiftRunner))

    def test_run_once(self):
        pass

    def test_run(self):
        pass


unittest.main(argv=[""], verbosity=2, exit=False)
