from unittest import TestCase


class TestTaxiEnv(TestCase):
    def test_map_at_location(self):
        from environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
        env = TaxiEnv()
        self.assertEqual('F', env.map_at_location([0, 2]))
