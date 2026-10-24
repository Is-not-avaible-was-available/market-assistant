from google.genai import types
from llm import ask

history = []
print("Market assistant ready. Type 'quit' to exit.")

while True:
    user_text = input("\nYou: ").strip()
    if user_text.lower() in ("quit", "exit"):
        break
    if not user_text:
        continue

    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    try:
        response = ask(history)
    except Exception:
        history.pop()          # remove the unanswered message so history stays valid
        print("Request failed. Try again later.")
        continue

    reply = response.text or ""
    history.append(types.Content(role="model", parts=[types.Part(text=reply)]))

    u = response.usage_metadata
    print(f"\nAssistant: {reply}")
    print(f"[tokens in={u.prompt_token_count} out={u.candidates_token_count} | messages={len(history)}]")