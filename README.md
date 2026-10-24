# Market Assistant

A command-line chatbot that answers questions about stocks using **tool calling** instead of memory. The LLM decides what to look up. Python fetches the data and does every calculation. Each answer is checked so that numbers the model states can be traced back to a tool result.

> Educational project. Not investment advice. Do not use it to make real trading decisions.

## Sample session

```
You: How has Infosys performed over the last 6 months, and how volatile is it?
  [tool] get_returns({'symbol': 'INFY.NS', 'period': '6mo'})
  [tool] get_volatility({'symbol': 'INFY.NS', 'period': '6mo'})

You: I bought 10 shares of Infosys at 1500. What is my profit or loss now?
  [tool] get_stock_price({'symbol': 'INFY.NS'})
  [tool] calculate({'expression': '(1153.8 - 1500) * 10'})

Assistant: ... Total Profit/Loss: -INR 3,462.00 ...
```

The model never states a price or does arithmetic on its own. If a ticker does not exist, it says so instead of inventing a price.

## How it works

```
user question
     |
     v
 agent.py  -- sends history + tool list --> Gemini (llm.py)
     ^                                          |
     |                              replies with a tool request
     |                                          |
     +-- runs the Python function (tools.py) <--+
     |
     +-- sends the result back, repeats until the model answers in text
     |
     v
 grounding check + audit log --> answer
```

| File | Purpose |
|---|---|
| `llm.py` | Gemini API helper with retries for server errors and per-minute limits. Stops immediately on daily-quota and request errors. |
| `tools.py` | Four tools: `get_stock_price`, `get_returns`, `get_volatility`, `calculate`. |
| `agent.py` | The tool-calling loop, system prompt, number grounding check, audit log. |
| `tests/` | Unit tests for the tools (no API calls needed). |
| `learning/` | Early exercises (chat loop, first tool call), kept for reference. |

## Design decisions

- **The LLM decides, Python computes.** Language models predict text. They do not calculate. Every number comes from a tool.
- **Safe calculator.** `calculate` parses expressions with Python's `ast` module and allows only numbers and arithmetic operators. It never uses `eval()` on model-generated text. Exponents are capped, and code-injection attempts are rejected (covered by tests).
- **Tool results carry context.** Each result includes the date of the data and a `note` with caveats, and the system prompt requires the model to report them.
- **Errors are returned, not raised.** Tools return `{"error": ...}` so the model can explain the problem instead of the program crashing.
- **Model-generated arguments are untrusted.** Tool calls are wrapped so a wrong or missing argument becomes an error message, not a crash.
- **Bounded loop.** The agent stops after a fixed number of tool steps so a confused model cannot loop forever.

## Guardrails

1. **Prompt rules:** always use tools for prices, never do arithmetic, always state currency and dates, report tool errors honestly.
2. **Number grounding check:** after each answer, every number is compared with the tool results and the user's own messages. Anything not found is flagged in the terminal. It is deliberately strict, so a legitimate rounding by the model also gets flagged.
3. **Step limit** on the tool loop.
4. **Audit log:** each question, tool call, tool result, and answer is appended to `audit.jsonl` (git-ignored) so any answer can be traced afterwards.

## Setup

Requires Python 3.9+ and a free Gemini API key from [Google AI Studio](https://aistudio.google.com).

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (it is git-ignored):

```
GEMINI_API_KEY=your-key-here
```

Run the assistant and the tests:

```bash
python agent.py
pytest -q
```

Model names change often. If you get a "model not found" error, change `MODEL` in `llm.py`.

## Known limitations

- **Data source:** `yfinance` is an unofficial scraper. It can break, be rate-limited, or return delayed data. Fine for learning, not for production.
- **Ticker lookup:** the model chooses tickers (such as `TCS.NS`) from its own knowledge. That works for large companies and may fail for obscure ones. A ticker-search tool would fix this.
- **Free-tier limits:** the Gemini free tier has low per-minute and per-day request limits, and each question costs 2 to 4 requests.
- **Grounding check is string-based:** it catches invented numbers, but it cannot tell whether a number is used in the correct context.
- **No tests for the agent loop yet:** the unit tests cover the tools. The loop was tested by hand.

## What I learned

- **A bug I almost shipped:** by default, `yfinance` returns dividend-adjusted historical prices. My first version reported a 6-month "price return" of -4.98% that actually included dividends, with a start price (1214.32) that was not the real closing price. With `auto_adjust=False`, the real start price was 1240.6 and the price return was -7.0%. The tool now reports both figures with accurate labels. Lesson: check tool output against a second source.
- **Models are stateless.** A chatbot "remembers" only because the code resends the whole conversation, and input tokens grow every turn.
- **Temperature 0 did not give identical outputs.** Hidden reasoning tokens vary between runs, so repeatability has to come from tools and checks, not from a temperature setting.
- **Rate limits shape design.** I hit per-minute and per-day quotas, and a retry helper that retried everything made it worse. It now retries only server errors and per-minute limits, and stops with the real error otherwise.
- **Models get retired.** A model I tried returned a 404 because it was no longer available to new accounts. Pin the model name in one place.
- **Tool descriptions are the model's only documentation.** Putting the `.NS` suffix rule in the description is what made it call `INFY.NS` correctly.

## Next

Project 2: RAG over company filings, with an evaluation set to measure answer quality.
