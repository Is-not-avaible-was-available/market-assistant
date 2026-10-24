from llm import ask
from google.genai import types

price_tool = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_stock_price",
        description="Get the latest closing price of a stock. NSE tickers end in .NS (e.g. INFY.NS), BSE in .BO, US tickers have no suffix (e.g. AAPL).",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol, e.g. INFY.NS or AAPL"}
            },
            "required": ["symbol"],
        },
    )
])

response = ask("What is Infosys trading at?", tools=[price_tool])

print("Text:", response.text)
print("Function calls:", response.function_calls)