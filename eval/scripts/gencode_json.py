import argparse
from pathlib import Path
from typing import Dict, Any, List
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

from scicode.parse.parse import (
    extract_function_name,
    get_function_from_code,
    read_from_jsonl
)
from scicode.gen.models import extract_python_script, get_model_function

DEFAULT_PROMPT_TEMPLATE = Path("eval", "data", "background_comment_template.txt").read_text()
BACKGOUND_PROMPT_TEMPLATE = Path("eval", "data", "multistep_template.txt").read_text()


class Gencode:
    def __init__(self, model: str, output_dir: Path,
                 prompt_dir: Path, with_background: bool, temperature: float,
                 num_samples: int=0, verifier: str=""):
        self.model = model
        self.output_dir = output_dir
        self.prompt_dir = prompt_dir
        self.with_background = with_background
        self.temperature = temperature
        self.num_samples = num_samples
        self.verifier = verifier
        self.previous_llm_code = []

    def _get_background_dir(self):
        return "with_background" if self.with_background else "without_background"

    def save_prompt_with_steps(self, prob_data: dict, prompt: str, num_steps: int) -> None:
        output_dir = Path(self.prompt_dir, Path(self.model).parts[-1], self._get_background_dir())
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / f"{prob_data['problem_id']}.{num_steps}.txt"
        output_file_path.write_text(prompt, encoding="utf-8")

    def save_response_with_steps(self, prob_data: dict, response: str,
                                 previous_code: str, num_steps: int, suffix: str = "") -> None:
        output_dir = (
                self.output_dir / Path(self.model).parts[-1] / self._get_background_dir()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        prob_id = prob_data["problem_id"]
        output_file_path = output_dir / f"{prob_id}.{num_steps}{suffix}.py"
        python_code = extract_python_script(response)
        output_file_path.write_text(f'{previous_code}\n{python_code}', encoding="utf-8")

    def generate_response_with_steps(
        self, prob_data: dict, num_steps: int, tot_steps: int, model="gpt-4o",
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            *, save: bool = True, num_repeats: int = 1, save_feedback_steps: bool = False) -> None:
        """

        Args:
            prob_data (dict): dict of the problem
            num_steps (int): Current generating step
            tot_steps (int): Total step of the problem
            model (str)
            prompt_template (str)
            save (bool, optional): Save propmt and model response. Defaults to True.
            num_repeat (int): Number of replicates for each subtask to save
        """
        prob_id = prob_data["problem_id"]
        output_file_path = (
                self.output_dir / Path(self.model).parts[-1] / self._get_background_dir()
                / f"{prob_id}.{num_steps}.py"
        )
        if num_steps == 1:
            self.previous_llm_code = [None] * tot_steps
        else:
            if len(self.previous_llm_code) != tot_steps:
                self.previous_llm_code = [None] * tot_steps
            for prev_step in range(num_steps - 1):
                if self.previous_llm_code[prev_step] is None:
                    if (prob_id == "13" and prev_step == 5) or (prob_id == "62" and prev_step == 0)\
                            or (prob_id == "76" and prev_step == 2):
                        prev_file_path = Path("eval", "data", f"{prob_id}.{prev_step+1}.txt")
                    else:
                        prev_file_path = (
                                self.output_dir / Path(self.model).parts[-1] / self._get_background_dir()
                                / f"{prob_id}.{prev_step + 1}.py"
                        )
                    if prev_file_path.is_file():
                        prev_file_content = prev_file_path.read_text(encoding='utf-8')
                        func_name = extract_function_name(prob_data["sub_steps"][prev_step]["function_header"])
                        function_code = get_function_from_code(prev_file_content, func_name)
                        self.previous_llm_code[prev_step] = function_code
                    else:
                        raise Exception(f'Generating {prob_id} step {num_steps} ahead of step {prev_step + 1}.')

        if output_file_path.exists():
            return
        prompt, previous_code, next_step_str = self.generate_prompt_with_steps(prob_data, num_steps, prompt_template)
        if save:
            self.save_prompt_with_steps(prob_data, prompt, num_steps)

        model_kwargs = {}
        model_kwargs["temperature"] = self.temperature
        if "claude" in model:
            model_kwargs["max_tokens"] = 4096
        elif "feedback" in model or "Feedback" in model:
            model_kwargs["code_prompt"] = next_step_str
        elif "sampling" in model:
            model_kwargs["code_prompt"] = next_step_str
            model_kwargs["num_samples"] = self.num_samples
            model_kwargs["verifier"] = self.verifier
        # write the response to a file if it doesn't exist
        model_fct = get_model_function(model, **model_kwargs)
        for index in range(num_repeats):
            response_from_llm = model_fct(prompt)
            if save_feedback_steps:
                final_response, all_responses = response_from_llm
                if index == 0:
                    self.previous_llm_code[num_steps - 1] = extract_python_script(final_response)
                for j, response in enumerate(all_responses):
                    suffix = f"_{j}"
                    self.save_response_with_steps(prob_data, response, previous_code, num_steps, suffix = suffix)
            else:
                if index == 0:
                    self.previous_llm_code[num_steps - 1] = extract_python_script(response_from_llm)
                suffix = f"_{index}"
                if num_repeats == 1:
                    suffix = ""
                self.save_response_with_steps(prob_data, response_from_llm, previous_code, num_steps, suffix = suffix)
    @staticmethod
    def process_problem_code(prob_data: dict, num_steps: int) -> str:
        header_docstring = prob_data['sub_steps'][num_steps - 1]['function_header']
        return_str = prob_data['sub_steps'][num_steps - 1]['return_line']
        string = f"{header_docstring}\n\n{return_str}"
        return string

    def process_problem_steps(self, problem_data: dict, num_steps: int):
        """Process problem data and return previous steps and next steps"""
        output_lines = []
        next_step = []
        previous_code = []
        for i in range(num_steps - 1):
            output_lines.append(problem_data["sub_steps"][i]["step_description_prompt"] + '\n' +
                                problem_data["sub_steps"][i]["step_background"] if self.with_background
                                else problem_data["sub_steps"][i]["step_description_prompt"])
            output_lines.append(self.previous_llm_code[i])
            previous_code.append(self.previous_llm_code[i])
            output_lines.append("------")

        next_step.append(problem_data["sub_steps"][num_steps - 1]["step_description_prompt"] + '\n' +
                         problem_data["sub_steps"][num_steps - 1]["step_background"] if self.with_background
                         else problem_data["sub_steps"][num_steps - 1]["step_description_prompt"])
        next_step.append(self.process_problem_code(problem_data, num_steps))
        output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
        next_step_str = "\n\n".join(next_step)
        previous_code_str = "\n".join(previous_code)
        return output_str, next_step_str, previous_code_str

    def generate_prompt_with_steps(self, prob_data: dict, num_steps: int,
                                   prompt_template=DEFAULT_PROMPT_TEMPLATE):
        # parse the input file and extract the content
        problem_steps_str, next_step_str, previous_code_str = self.process_problem_steps(prob_data,
                                                                                         num_steps)
        dependencies = prob_data["required_dependencies"]
        assert next_step_str
        return prompt_template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        ), f'{dependencies}\n{previous_code_str}\n', next_step_str


def get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model name"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results", "generated_code"),
        help="Output directory",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("eval", "data", "problems_all.jsonl"),
        help="Input directory",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path("eval_results", "prompt"),
        help="Prompt directory",
    )
    parser.add_argument(
        "--with-background",
        action="store_true",
        help="Include problem background if enabled",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Generation temperature",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help='Number of times to sample LLM for the repeated sampling methods'
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default="LLM_judge",
        help='Type of method to use to verify/grade generated samples'
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help='Number of replicates for each subtask'
    )
    parser.add_argument(
        "--save-feedback-steps",
        action="store_true",
        help="Save iteration steps (only relevant for feedback-based models)",
    )
    return parser


def process_problem(args: tuple) -> None:
    """
    Process a single problem with all its steps.
    
    Args:
        args: Tuple containing (
            problem: Dict containing problem data
            gcode_params: Dict containing parameters for Gencode initialization
            prompt_template: str for prompt template
            num_repeats: int for number of repeats
            save_feedback_steps: bool for saving feedback steps
        )
    """
    problem, gcode_params, prompt_template, num_repeats, save_feedback_steps = args
    print(f"Processing problem {problem['problem_id']}...")
    
    # Initialize Gencode instance for this process
    gcode = Gencode(
        model=gcode_params['model'],
        output_dir=gcode_params['output_dir'],
        prompt_dir=gcode_params['prompt_dir'],
        with_background=gcode_params['with_background'],
        temperature=gcode_params['temperature'],
        num_samples=gcode_params['num_samples'],
        verifier=gcode_params['verifier']
    )
    
    prob_id = problem['problem_id']
    steps = len(problem['sub_steps'])
    
    # Skip specific problem-step combinations
    skip_combinations = {
        ('13', 5),
        ('62', 0),
        ('76', 2)
    }
    
    for i in range(steps):
        if (prob_id, i) not in skip_combinations:
            try:
                gcode.generate_response_with_steps(
                    problem,
                    i + 1,
                    steps,
                    gcode_params['model'],
                    prompt_template,
                    num_repeats=num_repeats,
                    save_feedback_steps=save_feedback_steps
                )
            except Exception as e:
                print(f"Error processing problem {prob_id} step {i+1}: {str(e)}")

def main(
    model: str,
    output_dir: Path,
    input_path: Path,
    prompt_dir: Path,
    with_background: bool,
    temperature: float,
    num_samples: int,
    verifier: str,
    num_repeats: int,
    save_feedback_steps: bool,
    num_processes: int = None
) -> None:
    """
    Main function to process problems in parallel.
    """
    num_processes = multiprocessing.cpu_count()
    
    # Initialize shared parameters
    prompt_template = BACKGOUND_PROMPT_TEMPLATE if with_background else DEFAULT_PROMPT_TEMPLATE
    
    # Package Gencode parameters
    gcode_params = {
        'model': model,
        'output_dir': output_dir,
        'prompt_dir': prompt_dir,
        'with_background': with_background,
        'temperature': temperature,
        'num_samples': num_samples,
        'verifier': verifier
    }
    
    # Read data
    data = read_from_jsonl(input_path)
    
    # Prepare arguments for parallel processing
    process_args = [
        (problem, gcode_params, prompt_template, num_repeats, save_feedback_steps)
        for problem in data
    ]
    
    # Process problems in parallel with progress bar
    with Pool(processes=num_processes) as pool:
        list(tqdm(
            pool.imap(process_problem, process_args),
            total=len(process_args),
            desc="Processing problems"
        ))

if __name__ == "__main__":
    args = get_cli().parse_args()
    main(**vars(args))
