from functools import partial
from openai import OpenAI
from pathlib import Path
import anthropic
import google.generativeai as genai
import random
import string
import inspect
import ast
import config
import re
import os
import litellm
import subprocess
import sys
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
    #print("#######################################################################################")
    #print("PROMPT:")
    #print(final_prompt)
    response = generate_openai_response(final_prompt, model=model)
    #print("#######################################################################################")
    #print("RESPONSE:")
    #print(response)

    return response

def make_unit_tests(code_prompt, code, code_file_name, model="gpt-4-turbo-2024-04-09"):
    background_prompt = f"Your role is to write comprehensive unit tests for a function. You will receive the background for a python function as well as a potential implementation " \
                        "of the function. The implementation may be incorrect. Suggest unit tests that are comprehensive enough to find all of the potential pitfalls you think may happen." \
                        f"Return only the unit tests. Your response should only contain python code and in a form so that the unit tests can be run assuming there's a separate python file called {code_file_name} containing the function. Here is the original prompt used to generate the function: {code_prompt}" \
                        "Here is the returned implementation that you should write unit tests for: "
    unit_test_prompt = background_prompt + code
    unit_tests = generate_openai_response(unit_test_prompt, model=model)
    return unit_tests

def run_unit_tests(unit_tests, code_file_name, code, model="gpt-4-turbo-2024-04-09"):
    """
    Args:
        unit_tests (str): raw output from LLM containing the unit tests
        code_file_name (str): name of the file to run the code in
    """
    # Reformat python script
    python_script = unit_tests.split("```python")[1].split("```")[0] if '```python' in unit_tests else unit_tests.split('```')[1].split('```')[0]

    # Create temporary directory + files to run unit tests
    unitTest_file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15)) + ".py"
    output_dir = Path(os.getcwd()) / Path("temp_unit_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_unit_test_file = output_dir / unitTest_file_name
    temp_python_file = output_dir / code_file_name
    temp_unit_test_file.write_text(python_script, encoding="utf-8")

    sample_python_code = code.split("```python")[1].split("```")[0] if '```python' in code else code.split('```')[1].split('```')[0]
    temp_python_file.write_text(sample_python_code, encoding="utf-8")

    result = compiler_agent(temp_unit_test_file,model)

    Path.unlink(temp_python_file)
    Path.unlink(temp_unit_test_file)

    return result

def unit_test_agent(problem_prompt, code_prompt, samples_lst, model="gpt-4-turbo-2024-04-09"):
    """
    Args:
        problem_prompt (str): original prompt for the problem
        code_prompt (str): part of the original prompt describing what the code should do and outlining its skeleton
        samples_lst (lst): a list containing the sampled generated functions
        model (str): which LLM to use for writing the unit tests
    """
    code_file_name = "testingCode.py"
    summarize_unit_tests_prompt = "Your role is to summarize a set of unit tests. You will receive a set of unit test python files in the form (UNIT TESTS i): where i is the i-th set of unit tests. " \
                                  "Do not modify the unit tests, only remove redundant unit tests that are similar to one another and merge the rest into one python file. Only return the code, do not include " \
                                  "anything else in your response."
    
    max_prop_passed = 0
    best_code = samples_lst[0]
    unit_test_lst = []
    # Collect unit tests
    for sample_code in samples_lst:
        unit_tests = make_unit_tests(code_prompt, sample_code, code_file_name)
        unit_test_lst.append(unit_tests)

    # Summarize unit tests
    summarize_unit_tests_prompt += " ".join([f"(UNIT TESTS {i+1}): {x}" for i, x in enumerate(unit_test_lst)])
    all_unit_tests = generate_openai_response(summarize_unit_tests_prompt)

    # Test every sample code, keep the sample with the most unit tests passed
    for i, sample_code in enumerate(samples_lst):
        prop_tests_passed = run_unit_tests(all_unit_tests, code_file_name)['proportion_passed']
        print(f"Sample Code {i} passed {round(prop_tests_passed, 4) * 100}% of the unit tests")
        if prop_tests_passed > max_prop_passed:
            best_code = sample_code
            max_prop_passed = prop_tests_passed

    return best_code

def compiler_agent(python_file, model="gpt-4-turbo-2024-04-09"):
    """
    Args:
        python_file (str): the path to the python file being tested
    Returns:
        float: proportion of unit tests passed (0.0 to 1.0)
    """
    results = {
        'compilation_success': False,
        'proportion_passed': 0.0,
        'error_message': None,
        'output': None
    }

    def get_function_source(content: str, node: ast.FunctionDef) -> str:
        """Extract function source code using line numbers from AST node."""
        content_lines = content.splitlines()
        # Get the start and end lines, accounting for decorators
        start_line = min(d.lineno for d in node.decorator_list) - 1 if node.decorator_list else node.lineno - 1
        end_line = node.end_lineno
        
        # Extract and join the lines, preserving indentation
        function_lines = content_lines[start_line:end_line]
        return '\n'.join(function_lines)
    
    try:
        # Try to compile the Python file to check for syntax errors
        with open(python_file, 'r', encoding='utf-8') as file:
            file_content = file.read()
            compile(file_content, python_file, 'exec')
        results['compilation_success'] = True
        
        # Parse the file to extract test function source code
        tree = ast.parse(file_content)
        test_functions = {}
        
        # Extract all test functions from the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                # Get the source code for the function using line numbers
                function_body = get_function_source(file_content, node)
                test_functions[node.name] = function_body
        
        if not test_functions:
            print("Warning: No test functions found in the file")  # Debug output
        
        # Run the unit tests using subprocess with pytest's detailed output
        process = subprocess.run(
            [sys.executable, '-m', 'pytest', str(python_file), '-v'],
            capture_output=True,
            text=True
        )
        
        # Parse the test results and match with source code
        output_lines = process.stdout.split('\n')
        results['output'] = process.stdout
        total_tests = 0
        passed_tests = 0
        
        for line in output_lines:
            if '::test_' in line:
                total_tests += 1
                
                # Extract test name using regex
                match = re.search(r'::(test_\w+)', line)
                if match:
                    test_name = match.group(1)
                    test_code = test_functions.get(test_name, "Source code not found")
                    
                    test_result = {
                        'name': test_name,
                        'code': test_code,
                        'status': 'passed' if 'PASSED' in line else 'failed',
                        'error': None
                    }
                    
                    # If test failed, get the error message
                    if 'FAILED' in line:
                        # Find the error message in the following lines
                        error_index = output_lines.index(line)
                        error_message = []
                        i = error_index + 1
                        while i < len(output_lines) and output_lines[i].strip() and not output_lines[i].startswith('_'):
                            error_message.append(output_lines[i])
                            i += 1
                        test_result['error'] = '\n'.join(error_message).strip()
                    
                    results['test_details'].append(test_result)
                    
                    if 'PASSED' in line:
                        passed_tests += 1

        # Calculate proportion if there were any tests
        if total_tests > 0:
            results['proportion_passed'] = passed_tests / total_tests
            
    except Exception as e:
        results['error_message'] = str(e)
        
    return results

def aggregate_samples(problem_prompt, code_prompt, samples_lst, num_samples, verifier = "LLM_judge"):
    if verifier == "LLM_judge":
        return LLM_judge(problem_prompt, code_prompt, samples_lst, num_samples)
    elif verifier == "unit_test_agent":
        return unit_test_agent(problem_prompt, code_prompt, samples_lst)
    else:
        print("Not a recognized verifier")
        exit

def generate_openai_feedback_response(prompt: str, *, model, code_prompt: str = "",
                             temperature: float = 0, num_iterations: int = 3) -> str:
    current_code = generate_openai_response(prompt=prompt, temperature=temperature, model=model)
    code_file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15)) + ".py"
    unit_tests, output = "", ""
    all_codes = [current_code]
    for i in range(num_iterations):
        unit_test_prompt = f"Your role is to write new comprehensive unit tests for a function. You will receive four sections: (1) Code - this is the code being test (2) Old Unit Tests - these are the " \
                            "unit tests used in the past iteration, note that there may be errors in the unit tests and if blank then there are no current unit tests (3) Output - the result " \
                            "of running the old unit tests. If blank, there are no current unit tests (4) Original Prompt. Return a new set of unit tests that thoroughly test the code. Try to suggest unit tests " \
                            "that are distinct from the old unit tests. Only return the unit tests, do not return anything else. Your response should only contain python code and in a form so that the unit " \
                            f"tests can be run assuming there's a separate python file called {code_file_name} containing the function.\n\n" \
                            f"(1) Code: {current_code} \n\n" \
                            f"(2) Old Unit Tests: {unit_tests} \n\n" \
                            f"(3) Output: {output} \n\n" \
                            f"(4) Original Prompt: {code_prompt} \n\n"
        unit_tests = generate_openai_response(prompt=unit_test_prompt)
        output = run_unit_tests(unit_tests=unit_tests, code_file_name=code_file_name, code=current_code)['output']
        new_code_prompt = f"Your role is to modify a function so that it passes failed unit tests. You will receive four sections: (1) Code - this is the code being test (2) Unit Tests - these are the " \
                          "unit tests being used to test the code, note that there may be errors in the unit tests (3) Output - the result " \
                          "of running the unit tests. (4) Original Prompt. Return the modified code so it addresses the failed unit tests and " \
                          "any other errors you see. Only return the code and the commented background in the code, do not return anything else. Do not change any of the dependencies being "\
                          "installed, you are not allowed to use other python libraries other than those originally in the code.\n\n" \
                          f"(1) Code: {current_code} \n\n" \
                          f"(2) Unit Tests: {unit_tests} \n\n" \
                          f"(3) Output: {output} \n\n" \
                          f"(4) Original Prompt: {code_prompt} \n\n"
        current_code = generate_openai_response(prompt=new_code_prompt)
        all_codes.append(current_code)
    return current_code, all_codes

def generate_openai_sampledFeedback_response(prompt: str, *, model, code_prompt: str = "",
                             temperature: float = 0, num_iterations: int = 9) -> str:
    initial_code = generate_openai_response(prompt=prompt, temperature=temperature, model=model)
    all_codes = [initial_code]
    for i in range(num_iterations):
        implementations = " ".join([f"(Implementation {i+1}): {x}" for i, x in enumerate(all_codes)])
        distinct_code_prompt = f"Given a set of possible implementations for a function, your role is to design a new implementation that is distinct " \
                                "from this set of implementations. Make your implementation as different as possible while ensuring correctness. " \
                                "Do not return anything besides your implementation. Your response should only contain python code.\n\n" \
                                f"Task: {code_prompt} \n\n" \
                                f"Past implementations: {implementations}"
        new_code = generate_openai_response(prompt=distinct_code_prompt, temperature=temperature, model=model)
        all_codes.append(new_code)
    return all_codes[-1], all_codes
    

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
    if "samplingFeedback" in model:
        fct = generate_openai_sampledFeedback_response
        model=model.split("_")[0]
    elif "sampling" in model:
        fct = generate_openai_sampled_response
        model=model.split("_")[0]
    elif "feedback" in model:
        fct = generate_openai_feedback_response
        model=model.split("_")[0]
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

