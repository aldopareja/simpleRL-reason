## load the data

from datasets import load_dataset

data = load_dataset("json", data_files="/home/lab/simpleRL-reason/train/data/math_level3to5_data_processed_with_qwen_prompt.json", split="train")
data_bespoke = load_dataset("json", data_files="/new_data/oleg/datasets/bespoke-stratos-35k.jsonl", split="train")

## print a couple of examples

import random
import re

def convert_single_example(example: dict) -> list:
    """
    Convert a single data example into a list of messages.
    The example dictionary is expected to have a key 'input'
    which contains a string with segments like:
    
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
        ...
        <|im_end|>
        <|im_start|>assistant
    
    This function extracts the 'system' and 'user' segments,
    strips out the special tokens, and returns a list of messages
    suitable for a chat format.
    """
    text = example.get("input", "")
    
    # Regex to capture role and content between <|im_start|>...<|im_end|>
    pattern = r"<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    messages = []
    for role, content in matches:
        role = role.strip()
        content = content.strip()
        
        # Only collect 'system' and 'user' messages
        if role in ["system", "user"]:
            messages.append({
                "role": role,
                "content": content
            })
    
    example['messages'] = messages

    return example

print(convert_single_example(data[0]))
print(data[0]['input'])

data_with_messages = data.map(convert_single_example, num_proc=192)

## check if all data starts with `<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.`

def check_if_starts_with_system_prompt(example: dict) -> bool:
    example['starts_with_boxed']=example['input'].startswith("<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.")
    return example

print(sum(data.map(check_if_starts_with_system_prompt, num_proc=192)['starts_with_boxed'])/len(data))

print(data_bespoke[0])

## add the initial prompt for thinking to each user 

prompt_for_thinking = (
    "Your role as an assistant involves thoroughly exploring questions through a systematic "
    "long thinking process before providing the final precise and accurate solutions. This "
    "requires engaging in a comprehensive cycle of analysis, summarizing, exploration, "
    "reassessment, reflection, backtracing, and iteration to develop well-considered "
    "thinking process.\n\n"
    "Please structure your response into two main sections: Thought and Solution.\n\n"
    "In the Thought section, detail your reasoning process using the specified format:\n"
    "<|begin_of_thought|>\n"
    "{thought with steps separated with '\\n\\n'}\n"
    "<|end_of_thought|>\n\n"
    "Each step should include detailed considerations such as analisying questions, "
    "summarizing relevant findings, brainstorming new ideas, verifying the accuracy "
    "of the current steps, refining any errors, and revisiting previous steps.\n\n"
    "In the Solution section, based on various attempts, explorations, and reflections "
    "from the Thought section, systematically present the final solution that you deem "
    "correct. The solution should remain a logical, accurate, concise expression style "
    "and detail necessary step needed to reach the conclusion, formatted as follows:\n"
    "<|begin_of_solution|>\n"
    "{final formatted, precise, and clear solution}\n"
    "<|end_of_solution|>\n\n"
    "Now, try to solve the following question through the above guidelines:\n"
    "Return your final response within \\boxed{}. "
)

def add_prompt_for_thinking(example: dict) -> dict:
    assert example['messages'][1]['role'] == 'user'
    example['messages'][1]['content'] = prompt_for_thinking + example['messages'][1]['content']
    return example

data_with_prompt_for_thinking = data_with_messages.map(add_prompt_for_thinking, num_proc=192)

print(data_with_prompt_for_thinking[0]['messages'][1]['content'])

## change the system prompt for the granite one 

granite_system_prompt = 'I am a Red Hat® Instruct Model, an AI language model developed by Red Hat and IBM Research based on the granite-3.1-8b-base model. My primary role is to serve as a chat assistant.'

def change_system_prompt(example: dict) -> dict:
    example['messages'][0]['content'] = granite_system_prompt
    return example

data_with_granite_system_prompt = data_with_prompt_for_thinking.map(change_system_prompt, num_proc=192)

[print(m['content']) for m in data_with_granite_system_prompt[0]['messages']]

## save the data

data_with_granite_system_prompt.to_json("/home/lab/simpleRL-reason/train/data/math_level3to5_data_processed_with_qwen_prompt_with_messages_granite_system_prompt.jsonl", lines=True)

## load a bespoke model with reasoning and see what it does

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /new_data/experiments_rh/granite-r1-bespoke-v8/hf_format/samples_2761884 --tensor-parallel-size 8
'''


## process with the models tokenizer

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/new_data/experiments_rh/granite-r1-bespoke-v8/hf_format/samples_2761884")

print(tokenizer.apply_chat_template(data_with_granite_system_prompt[0]['messages'], tokenize=False, add_generation_prompt=True))

def make_input_using_tokenizer(example: dict) -> dict:
    example['input'] = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=True)
    return example

data_with_granite_system_prompt_tokenized = data_with_granite_system_prompt.map(make_input_using_tokenizer, num_proc=192)

print(data_with_granite_system_prompt_tokenized[0]['input'])
data_with_granite_system_prompt_tokenized.to_json("/home/lab/simpleRL-reason/train/data/math_level3to5_data_processed_with_qwen_prompt_with_messages_granite_system_prompt_tokenized.jsonl", lines=True)

## 

import aiohttp
import asyncio
import json
from tqdm.asyncio import tqdm_asyncio  # Import tqdm for asyncio

# Create a semaphore to limit concurrent requests
CONCURRENT_REQUESTS = 192  # Adjust this number based on your needs
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

async def query_vllm_server(example: dict, host: str = "http://localhost:8000/v1/chat/completions") -> dict:
    """
    Send an async request to the VLLM server and get the completion response.
    Uses a semaphore to limit concurrent requests.
    """
    async with semaphore:  # This will wait if we've hit the concurrent request limit
        async with aiohttp.ClientSession() as session:
            request_data = {
                "model": "/new_data/experiments_rh/granite-r1-bespoke-v8/hf_format/samples_2761884",
                "messages": example['messages'],
                "max_tokens": 8192,
                "temperature": 0.7,
                "stream": False
            }
            
            async with session.post(
                host,
                json=request_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                result = (await response.json())
                example['full_output'] = result
                example['model_answer'] = result['choices'][0]['message']['content']
                return example

# Example of processing multiple examples concurrently with progress bar
async def process_batch(examples):
    # Using tqdm_asyncio.gather to show progress
    return await tqdm_asyncio.gather(
        *[query_vllm_server(example) for example in examples],
        desc="Processing examples"
    )

# Run with:
results = asyncio.run(process_batch(data_with_granite_system_prompt_tokenized.select(range(100))))   

print(results)

## print the results

for result in results[:10]:
    print(result)
    print('\033[92m' + '-'*100 + '\033[0m')


'''
curl http://10.241.128.36:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/new_data/experiments_rh/granite-r1-bespoke-v8/hf_format/samples_2761884",
    "messages": [
      {"role": "system", "content": "I am a Red Hat® Instruct Model"},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 8192,
    "temperature": 0.7,
    "stream": false
  }'
'''
