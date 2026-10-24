import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
client = genai.Client()

MODEL = "gemini-3.1-flash-lite"   # change this one line if a model is unavailable

SYSTEM = "You are a concise financial markets assistant. Explain things simply."


def ask(question, system=SYSTEM, temperature=1.0, max_tokens=1000, tools=None):
    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    tools=tools,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except errors.ServerError as e:              # 5xx: Google is busy, retry
            wait = 3 * 2 ** attempt
            print(f"Server error {e.code}, retrying in {wait}s...")
            time.sleep(wait)
        except errors.ClientError as e:
            if e.code == 429 and "PerDay" not in str(e):   # per-minute limit
                print("Per-minute limit, waiting 45s...")
                time.sleep(45)
            else:                                    # daily limit, 404, bad request
                print("Stopping. Real error:\n", e)
                raise
    raise RuntimeError("Still failing. Try later or change MODEL.")


def show(response):
    u = response.usage_metadata
    print(response.text)
    print(f"[in={u.prompt_token_count} out={u.candidates_token_count} "
          f"thinking={u.thoughts_token_count} "
          f"finish={response.candidates[0].finish_reason}]")


if __name__ == "__main__":
    show(ask("What is Apple's stock price right now?"))