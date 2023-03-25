import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="davinci-instruct-beta",
  prompt="Correct this English text: Today I have went to the store to to buys some many bottle of water.\n\nToday I have gone to the store to buy some water bottles.",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)