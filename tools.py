import ast
import logging
import math
import operator
import warnings
 
import numpy as np
import yfinance as yf
from google.genai import types
 
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
 
VALID_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
 
 
# ---------------------------------------------------------------------------
# Market data tools
# ---------------------------------------------------------------------------
 
def get_stock_price(symbol: str) -> dict:
    """Return the latest closing price for a ticker, e.g. INFY.NS or AAPL."""
    try:
        ticker = yf.Ticker(symbol)
        # auto_adjust=False -> "Close" is the real closing price, not dividend-adjusted
        hist = ticker.history(period="5d", auto_adjust=False)
        if hist.empty:
            return {"error": f"No data for '{symbol}'. NSE stocks need .NS, e.g. INFY.NS"}
        try:
            currency = ticker.fast_info["currency"]
        except Exception:
            currency = "unknown"
        return {
            "symbol": symbol.upper(),
            "price": round(float(hist["Close"].iloc[-1]), 2),
            "currency": currency,
            "as_of": str(hist.index[-1].date()),
            "note": "Latest available close; data may be delayed.",
        }
    except Exception as e:
        return {"error": f"Lookup failed: {e}"}
 
 
def _history(symbol, period):
    if period not in VALID_PERIODS:
        return None, {"error": f"period must be one of {VALID_PERIODS}"}
    hist = yf.Ticker(symbol).history(period=period, auto_adjust=False)
    if hist.empty or len(hist) < 2:
        return None, {"error": f"Not enough data for '{symbol}'. NSE stocks need .NS"}
    return hist, None
 
 
def get_returns(symbol: str, period: str = "1y") -> dict:
    """Price return and dividend-inclusive return of a stock over a period."""
    try:
        hist, err = _history(symbol, period)
        if err:
            return err
        start, end = float(hist["Close"].iloc[0]), float(hist["Close"].iloc[-1])
        result = {
            "symbol": symbol.upper(),
            "period": period,
            "start_date": str(hist.index[0].date()),
            "start_price": round(start, 2),
            "end_date": str(hist.index[-1].date()),
            "end_price": round(end, 2),
            "price_return_pct": round((end / start - 1) * 100, 2),
        }
        if "Adj Close" in hist.columns:
            adj_start = float(hist["Adj Close"].iloc[0])
            adj_end = float(hist["Adj Close"].iloc[-1])
            result["return_incl_dividends_pct"] = round((adj_end / adj_start - 1) * 100, 2)
            result["note"] = (
                "price_return_pct uses actual closing prices; "
                "return_incl_dividends_pct assumes dividends were reinvested."
            )
        else:
            result["note"] = "Price return only. Dividend-adjusted data was not available."
        return result
    except Exception as e:
        return {"error": f"Lookup failed: {e}"}
 
 
def get_volatility(symbol: str, period: str = "1y") -> dict:
    """Annualized volatility and maximum drawdown of a stock over a period."""
    try:
        hist, err = _history(symbol, period)
        if err:
            return err
        close = hist["Close"]
        daily = np.log(close).diff().dropna()
        vol = float(daily.std() * math.sqrt(252))  # 252 trading days per year
        max_dd = float((close / close.cummax() - 1).min())
        return {
            "symbol": symbol.upper(),
            "period": period,
            "annualized_volatility_pct": round(vol * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "trading_days_used": len(daily),
            "note": "Based on daily log returns of closing prices.",
        }
    except Exception as e:
        return {"error": f"Lookup failed: {e}"}
 
 
# ---------------------------------------------------------------------------
# Safe calculator: never use eval() on text produced by an LLM
# ---------------------------------------------------------------------------
 
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}
 
 
def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Only numbers and + - * / % ** ( ) are allowed")
 
 
def calculate(expression: str) -> dict:
    """Evaluate arithmetic exactly, e.g. '(1153.8 - 1500) * 10'."""
    try:
        tree = ast.parse(expression, mode="eval")
        return {"expression": expression, "result": round(_eval(tree.body), 6)}
    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except Exception as e:
        return {"error": f"Cannot calculate: {e}"}
 
 
# ---------------------------------------------------------------------------
# Tool declarations: this is all the model ever sees of our functions
# ---------------------------------------------------------------------------
 
def _decl(name, description, properties, required):
    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters={"type": "object", "properties": properties, "required": required},
    )
 
 
SYMBOL = {
    "type": "string",
    "description": "Ticker. NSE ends in .NS (INFY.NS), BSE .BO, US has no suffix (AAPL)",
}
PERIOD = {
    "type": "string",
    "description": "One of: 1mo, 3mo, 6mo, 1y, 2y, 5y. Default 1y",
}
 
finance_tools = types.Tool(function_declarations=[
    _decl(
        "get_stock_price",
        "Latest closing price of a stock.",
        {"symbol": SYMBOL},
        ["symbol"],
    ),
    _decl(
        "get_returns",
        "Percentage return of a stock over a period: price return and return "
        "including dividends, with start and end prices and dates.",
        {"symbol": SYMBOL, "period": PERIOD},
        ["symbol"],
    ),
    _decl(
        "get_volatility",
        "Annualized volatility and maximum drawdown of a stock over a period.",
        {"symbol": SYMBOL, "period": PERIOD},
        ["symbol"],
    ),
    _decl(
        "calculate",
        "Exact arithmetic. Use for ANY calculation such as profit, percentages, "
        "totals. Example: '(1153.8 - 1500) * 10'.",
        {"expression": {
            "type": "string",
            "description": "Arithmetic expression using numbers and + - * / ** ( )",
        }},
        ["expression"],
    ),
])
 
TOOL_FUNCTIONS = {
    "get_stock_price": get_stock_price,
    "get_returns": get_returns,
    "get_volatility": get_volatility,
    "calculate": calculate,
}
 
 
# ---------------------------------------------------------------------------
# Quick manual test (uses no Gemini quota): python tools.py
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    print(get_stock_price("INFY.NS"))
    print(get_returns("INFY.NS", "6mo"))
    print(get_volatility("INFY.NS", "1y"))
    print(calculate("(1153.8 - 1500) * 10"))
    print(calculate("__import__('os').system('echo hacked')"))  # must be rejected
    print(get_returns("INFY.NS", "10y"))               