from functools import partial
from openai import OpenAI
import anthropic
import google.generativeai as genai
import random
import string
import config
import re
import os
import litellm
from litellm.utils import validate_environment as litellm_validate_environment

from scicode import keys_cfg_path
from scicode.utils.log import get_logger

logger = get_logger("models")


def get_config():
    if not keys_cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {keys_cfg_path}")
    return config.Config(str(keys_cfg_path))

def generate_litellm_response(prompt: str, *, model: str, **kwargs) -> str:
    """Call the litellm api to generate a response"""
    # litellm expects all keys as env variables
    config = get_config()
    for key, value in config.as_dict().items():
        if key in os.environ and os.environ[key] != value:
            logger.warning(f"Overwriting {key} from config with environment variable")
        else:
            os.environ[key] = value
    # Let's validate that we have everythong for this model
    env_validation = litellm_validate_environment(model)
    if not env_validation.get("keys_in_environment") or env_validation.get("missing_keys", []):
        msg = f"Environment validation for litellm failed for model {model}: {env_validation}"
        raise ValueError(msg)
    response = litellm.completion(
        model=model,
        messages = [
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )
    return response.choices[0].message.content

def generate_openai_response(prompt: str, *, model="gpt-4-turbo-2024-04-09",
                             temperature: float = 0) -> str:
    """call the openai api to generate a response"""
    key: str = get_config()["OPENAI_KEY"]  # type: ignore
    client = OpenAI(api_key=key)
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content

def LLM_judge(problem_prompt, code_prompt, samples_lst, num_samples, model="gpt-4-turbo-2024-04-09"):
    # Aggregate samples so that they follow the form "(i): sample i response"
    samples_string = " ".join([f"(EXPERT {i+1} RESPONSE): {x}" for i, x in enumerate(samples_lst)])

    final_prompt = f"Your role is a supervisor. You receive {num_samples} different responses, each from an expert. Each response is separated in the form " \
                         "(EXPERT i RESPONSE): where i is the ID of the expert and the contents proceeding that is the response for that ID. Your job is " \
                         "to reach a final answer based off of these responses. When there are disagreements, use your best judgment to " \
                         f"resolve them. Answer only to the original prompt, do not include any other statements. Here is the original prompt: " \
                         f"{code_prompt}. Here are the experts' responses: {samples_string}"
    print("#######################################################################################")
    print("PROMPT:")
    print(final_prompt)
    response = generate_openai_response(final_prompt, model=model)
    print("#######################################################################################")
    print("RESPONSE:")
    print(response)

    return response

def unit_test_agent(problem_prompt, samples_lst, model="gpt-4-turbo-2024-04-09"):
    background_prompt = f"Your role is to write comprehensive unit tests for code. You will receive the background for the code as well as the code's skeleton." \
                          "Return solely the unit tests so that they can be run from a different file in the same directory as the code."
    temp_python_file = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15)) + ".py"
    unit_test_prompt = background_prompt + problem_prompt
    unit_tests = generate_openai_response(unit_test_prompt, model=model)
    #print(unit_test_prompt)
    #print("##################################################################")
    #print(unit_tests)

    '''
    for code_sample in samples_lst:
        temp_python_file = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15)) + ".py"
        python_code = extract_python_script(code_sample)
        temp_python_file.write_text(python_code, encoding="utf-8")
        sample_prompt = f"{background_prompt} {code_sample}"
        unit_tests = generate_openai_response(sample_prompt, model=model)
        passed = compiler_agent(temp_python_file, unit_tests)
    ''' 

    def compiler_agent(python_file, unit_tests, model="gpt-4-turbo-2024-04-09"):
        """
        Args:
            python_file (str): the path to the python file being tested
            unit_tests (str): the path to the unit tests
        Returns a summary of the performance of the unit tests
        """
        pass

def aggregate_samples(problem_prompt, code_prompt, samples_lst, num_samples, verifier = "LLM_judge"):
    if verifier == "LLM_judge":
        return LLM_judge(problem_prompt, code_prompt, samples_lst, num_samples)
    elif verifier == "unit_test_agent":
        return unit_test_agent(problem_prompt, samples_lst)
    else:
        print("Not a recognized verifier")
        exit


def generate_openai_sampled_response(prompt: str, *, model, code_prompt: str = "",
                             temperature: float = 0.7, num_samples: int = 3, verifier: str = "LLM_judge") -> str:
    """call the openai api to generate a response"""
    sampling_temperature = 0.7 #### MAKE THIS AN ARG LATER
    key: str = get_config()["OPENAI_KEY"]  # type: ignore
    client = OpenAI(api_key=key)
    completion = client.chat.completions.create(
        model=model,
        temperature=sampling_temperature,
        n=num_samples,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    all_responses_lst = [choice.message.content for choice in completion.choices]
    final_response = aggregate_samples(prompt, code_prompt, all_responses_lst, num_samples, verifier=verifier)

    return final_response


def generate_anthropic_response(prompt, *, model="claude-3-opus-20240229",
                                max_tokens: int = 4096, temperature: float = 0) -> str:
    """call the anthropic api to generate a response"""
    key: str = get_config()["ANTHROPIC_KEY"]  # type: ignore
    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return message.content[0].text


def generate_google_response(prompt: str, *, model: str = "gemini-pro",
                             temperature: float = 0) -> str:
    """call the api to generate a response"""
    key: str = get_config()["GOOGLE_KEY"]  # type: ignore
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name=model)
    response = model.generate_content(prompt,
                                      generation_config=genai.GenerationConfig(temperature=temperature),
                                      # safety_settings=[
                                      #     {
                                      #         "category": "HARM_CATEGORY_HARASSMENT",
                                      #         "threshold": "BLOCK_NONE",
                                      #     },
                                      #     {
                                      #         "category": "HARM_CATEGORY_HATE_SPEECH",
                                      #         "threshold": "BLOCK_NONE",
                                      #     },
                                      #     {
                                      #         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                      #         "threshold": "BLOCK_NONE",
                                      #     },
                                      #     {
                                      #         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                      #         "threshold": "BLOCK_NONE"
                                      #     }
                                      # ]
                                      )
    try:
        return response.text
    except ValueError:
        print(f'prompt:\n{prompt}')
        # If the response doesn't contain text, check if the prompt was blocked.
        print(f'prompt feedback:\n{response.prompt_feedback}')
        # Also check the finish reason to see if the response was blocked.
        print(f'finish reason:\n{response.candidates[0].finish_reason.name}')
        # If the finish reason was SAFETY, the safety ratings have more details.
        print(f'safety rating:\n{response.candidates[0].safety_ratings}')
        raise ValueError("Generate response failed.")


def get_model_function(model: str, **kwargs):
    """Return the appropriate function to generate a response based on the model"""
    if "sampling" in model:
        fct = generate_openai_sampled_response
        model=model.replace("_sampling", "")
    elif model.startswith("litellm/"):
        model = model.removeprefix("litellm/")
        fct = generate_litellm_response
    elif "gpt" in model:
        fct = generate_openai_response
    elif "claude" in model:
        fct = generate_anthropic_response
    elif "gemini" in model:
        fct = generate_google_response
    elif model == "dummy":
        fct = generate_dummy_response
    else:
        raise ValueError(f"Model {model} not supported")
    return partial(fct, model=model, **kwargs)


def generate_dummy_response(prompt: str, **kwargs) -> str:
    """Used for testing as a substitute for actual models"""
    return "Blah blah\n```python\nprint('Hello, World!')\n```\n"


def extract_python_script(response: str):
    # We will extract the python script from the response
    if '```' in response:
        python_script = response.split("```python")[1].split("```")[0] if '```python' in response else response.split('```')[1].split('```')[0]
    else:
        print("Fail to extract python code from specific format.")
        python_script = response
    python_script = re.sub(r'^\s*(import .*|from .*\s+import\s+.*)', '', python_script, flags=re.MULTILINE)
    return python_script

