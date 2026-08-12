import os

# Run this example on a single GPU
os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["ROCM_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer

# this example requires the LLama 3.1 8B Instruct model
# Provided that you have a HF token to access the Llama models, you can download it with
# huggingface-cli download --token <your_token> meta-llama/Llama-3.1-8B-Instruct --local-dir Llama-3.1-8B-Instruct

# either set this environment variable before running the example, or adapt the path
model_id = os.getenv("LLAMA3_8B_PATH", "/storage/models/Llama-3.1-8B-Instruct/")

## Load the original model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

print("Tokenizer class: ", tokenizer.__class__)
print("Chat template: ", tokenizer.chat_template)
print("---")


# 5 tokens prompt
prompt = "Hello my name is Ashley"

inputs = tokenizer(prompt, return_tensors="pt")

# direct application
print("---")
print("Prompt: ", prompt)
print("Tokenized input ids: ", inputs["input_ids"])
print("---")


# chat template application
messages = [
    {"role": "user", "content": prompt},
]

chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

print("---")
print("Chat template prompt: ", chat_prompt)
print("---")

# chat template application + add generation prompt
messages = [
    {"role": "user", "content": prompt},
]

chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("---")
print("Chat template prompt + generation prompt: ", chat_prompt)
print("---")

# chat template application + continue last message
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": "<thoughts>"}
]

chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)

print("---")
print("Chat template prompt + continue last message: ", chat_prompt)
print("---")

#
# tools
#

# First, define a tool
def get_current_temperature(location: str) -> float:
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 22.  # A real function should probably actually get the temperature!

# Next, create a chat and apply the chat template
messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]

chat_prompt = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], tokenize=False, tools_in_user_message=False)

print("---")
print("Chat template prompt + tools: ", chat_prompt)
print("---")

# With a tool call
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France"}}
messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"},
  {"role": "assistant", "content": "Let's call the tool get_current_temperature('Paris, France')", "tool_calls": [{"type": "function", "function": tool_call}]},
  {"role": "tool", "content": "22"}
]

# tools_in_user_message=True is better for KV-caching
chat_prompt = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], tokenize=False, tools_in_user_message=True)

print("---")
print("Chat template prompt + tool call: ", chat_prompt)
print("---")

###########################
# Use the thinking template
###########################

# read content from file
def read_file_content(file_path: str) -> str:
    """
    Read the content of a file and return it as a string.
    
    Args:
        file_path: The path to the file to read.
    Returns:
        The content of the file as a string.
    """
    with open(file_path, 'r') as file:
        return file.read()

think_chat_template = read_file_content("examples/thinking_chat_template.jinja")

tokenizer.chat_template = think_chat_template

print("Thinking chat template: ", tokenizer.chat_template)

#
# tools
#

# Next, create a chat and apply the chat template
messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries."},
  {"role": "user", "content": "Hey, what's the temperature in Paris and Lyon right now?"}
]

chat_prompt = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], tokenize=False, add_generation_prompt=True)

print("---")
print("Thinking chat template prompt + tools: ", chat_prompt)
print("---")

# With a tool call
tool_call_paris = {"name": "get_current_temperature", "arguments": {"location": "Paris, France"}}
tool_call_lyon = {"name": "get_current_temperature", "arguments": {"location": "Lyon, France"}}

messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries."},
  {"role": "user", "content": "Hey, what's the temperature in Paris and Lyon right now?"},
  {
      "role": "assistant",
      "reasoning_content": "I can call the tool get_current_temperature for both Paris and Lyon to get the current temperatures.",
      "content": "Let's call the tool get_current_temperature('Paris, France') and get_current_temperature('Lyon, France')",
      "tool_calls": [{"type": "function", "function": tool_call_paris}, {"type": "function", "function": tool_call_lyon}]},
  {"role": "tool", "content": [22, 18]},
  {
      "role": "assistant",
      "reasoning_content": "The tool results indicate that the current temperature in Paris is 22°C and in Lyon is 18°C.",
      "content": "The current temperature in Paris is 22°C and in Lyon is 18°C."
  },
  {
      "role": "user",
      "content": "Thanks! Can you also tell me the temperature in Marseille?"
  },
]

# tools_in_user_message=True is better for KV-caching
chat_prompt = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], tokenize=False, add_generation_prompt=True)

print("---")
print("Thinking chat template prompt + tool call: ", chat_prompt)
print("---")


