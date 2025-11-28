from llm_client import call_llm

resp = call_llm("Reply with only the number: 2 + 3 = ?")
print("LLM response:", repr(resp))
