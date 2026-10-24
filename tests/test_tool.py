import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools import calculate, get_returns


def test_calculate_basic():
    assert calculate("(1153.8 - 1500) * 10")["result"] == -3462.0

def test_calculate_rejects_code():
    assert "error" in calculate("__import__('os').system('echo hi')")

def test_calculate_divide_by_zero():
    assert "error" in calculate("1/0")

def test_calculate_huge_exponent():
    assert "error" in calculate("9**999")

def test_bad_period_rejected_before_network():
    assert "error" in get_returns("INFY.NS", "10y")