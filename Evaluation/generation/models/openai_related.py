from openai import OpenAI

openai_api_key = "REPLACE WITH YOUR OPENAI API KEY"

def gpt_generation(prompt,model,temperature=0,max_tokens=1024,top_p=1,frequency_penalty=0,presence_penalty=0):
    client = OpenAI(api_key = openai_api_key)
    conversation = [{'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(
      model=model,
      messages=conversation,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty
    )
    return response.choices[0].message.content, (response.usage.prompt_tokens, response.usage.completion_tokens)


def gpt_generation_multi_turn(prompt,model,temperature=0,max_tokens=1024,top_p=1,frequency_penalty=0,presence_penalty=0):
    if type(prompt) is not dict or 'conversation' not in prompt.keys():
        raise ValueError("For multiple turn case, the prompt dict provided is not in correct format")
    client = OpenAI(api_key = openai_api_key)
    conversation = prompt['conversation']
    response = client.chat.completions.create(
      model=model,
      messages=conversation,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty
    )
    return response.choices[0].message.content, (response.usage.prompt_tokens, response.usage.completion_tokens)
  
