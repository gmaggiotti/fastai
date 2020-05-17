from unittest import TestCase
from unittest.mock import patch

class Calculator:
    def sum(self, a, b):
        return a + b

class TestCalc(Calculator):
    def sum(self, a,b):
        return 6

class TestCalculator(TestCase):
    def setUp(self):
        self.calc = TestCalc()

    def test_sum(self):
        answer = self.calc.sum(2, 4)
        self.assertEqual(answer, 6)

class TestCalculator1(TestCase):
    @patch('TestCalculator.Calculator.sum', return_value=9)

    def test_sum1(self, sum):
        self.assertEqual(sum(2,3), 9)