import datetime
import json
import re
from google.genai import types
from llm import ask
from tools import finance_tools, TOOL_FUNCTIONS

SYSTEM = """You are a careful financial markets assistant.
Rules:
- Never state prices, returns, or volatility from memory. Always use the tools.
- If the user asks for the current or latest price, call get_stock_price again, even if you reported one earlier.
- Never do arithmetic yourself. Use the calculate tool for every calculation.
- Always mention the currency and the dates behind any number you report.
- Mention any caveat in a tool's 'note' field.
- If a tool returns an error, tell the user plainly. Do not guess a number.
- This is educational information, not investment advice."""

MAX_STEPS = 6
NUM = re.compile(r"\d[\d,]*\.?\d*")


def numbers_in(text):
    out = set()
    for m in NUM.findall(text):
        try:
            out.add(float(m.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def ungrounded_numbers(answer, sources):
    allowed = set()
    for s in sources:
        allowed |= numbers_in(s)
    return sorted(n for n in numbers_in(answer)
                  if not any(abs(n - a) <= 0.011 for a in allowed))


def audit(entry):
    entry["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open("audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_turn(history, turn_log):
    for step in range(MAX_STEPS):
        response = ask(history, system=SYSTEM, tools=[finance_tools])
        calls = response.function_calls
        if not calls:
            return response

        history.append(response.candidates[0].content)
        result_parts = []
        for call in calls:
            args = dict(call.args)
            print(f"  [tool] {call.name}({args})")
            fn = TOOL_FUNCTIONS.get(call.name)
            try:
                result = fn(**args) if fn else {"error": f"Unknown tool {call.name}"}
            except TypeError as e:
                result = {"error": f"Bad arguments: {e}"}
            turn_log.append({"tool": call.name, "args": args, "result": result})
            result_parts.append(
                types.Part.from_function_response(name=call.name, response=result)
            )
        history.append(types.Content(role="user", parts=result_parts))

    raise RuntimeError("Too many tool steps without a final answer")


history = []
sources = []   # everything the answer's numbers are allowed to come from
print("Market assistant ready. Type 'quit' to exit.")

while True:
    text = input("\nYou: ").strip()
    if text.lower() in ("quit", "exit"):
        break
    if not text:
        continue

    sources.append(text)
    start = len(history)
    turn_log = []
    history.append(types.Content(role="user", parts=[types.Part(text=text)]))

    try:
        response = run_turn(history, turn_log)
    except Exception as e:
        del history[start:]
        print("Request failed:", e)
        continue

    answer = response.text or "(empty reply)"
    history.append(response.candidates[0].content)
    sources.extend(json.dumps(t["args"]) + " " + json.dumps(t["result"]) for t in turn_log)

    bad = ungrounded_numbers(answer, sources)
    print("\nAssistant:", answer)
    if bad:
        print(f"\n  [check] Numbers not found in tool results or your messages: {bad}")

    audit({"question": text, "tools": turn_log, "answer": answer,
           "unverified_numbers": bad,
           "tokens_in": response.usage_metadata.prompt_token_count})