from pathlib import Path
import json
import subprocess
import time
import shutil
import numpy as np
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt

from scicode.parse.parse import H5PY_FILE
from scicode.parse.parse import read_from_jsonl


PROB_NUM = 80
DEV_PROB_NUM = 15
STEP_NUM = 288
DEV_STEP_NUM = 50


def _get_background_dir(with_background):
    return "with_background" if with_background else "without_background"

def test_code_with_iterations(
    model_name: str,
    code_dir: str,
    log_dir: str,
    output_dir: str,
    jsonl_path: str,
    target_iteration: int = 3,
    dev_set: bool = False,
    with_background: bool = False
) -> None:
    """
    Test code with multiple iterations and analyze performance across iterations.
    
    Args:
        model_name: Name of the model
        code_dir: Directory containing code files
        log_dir: Directory for logs
        output_dir: Directory for output
        jsonl_path: Path to JSONL data
        max_iterations: Maximum number of iterations to analyze
        target_iteration: Iteration to use for final results
        dev_set: Whether this is dev set
        with_background: Whether background is used
    """
    jsonl_data = read_from_jsonl(jsonl_path)
    json_dct = {}
    json_idx = {}
    
    for prob_data in jsonl_data:
        json_dct[prob_data['problem_id']] = len(prob_data['sub_steps'])
        json_idx[prob_data['problem_id']] = jsonl_data.index(prob_data)
    
    start_time = time.time()
    code_dir_ = Path(code_dir, model_name, _get_background_dir(with_background))
    tmp_dir = Path(f'tmp_{start_time}')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Track results for each iteration
    iteration_results = defaultdict(lambda: {
        'correct_problems': np.zeros(PROB_NUM),
        'total_problems': np.zeros(PROB_NUM),
        'correct_steps': set(),
        'correct_dict': defaultdict(list),
        'failed_dict': defaultdict(list)
    })
    
    # Process and group files by iteration
    files_by_iteration = defaultdict(list)
    for file_path in code_dir_.iterdir():
        if not file_path.is_file():
            continue
            
        file_name = file_path.stem
        try:
            # Parse filename structure: problem_id.step_iteration.py
            prob_id, rest = file_name.split(".")
            step, iteration = rest.split("_")
            iteration = int(iteration)
            files_by_iteration[iteration].append((file_path, prob_id, step))
        except ValueError:
            print(f"Skipping malformed filename: {file_name}")
            continue
    
    def run_script(script_path: Path) -> Tuple[int, str]:
        """Run a script and return status code and output."""
        try:
            result = subprocess.run(
                ['python', str(script_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=180
            )
            return 0, result.stdout
        except subprocess.CalledProcessError as e:
            return 1, e.output
        except subprocess.TimeoutExpired as e:
            return 2, str(e)
    
    # Process each iteration
    print(files_by_iteration[1])
    for iteration in range(0, target_iteration + 1):
        results = iteration_results[iteration]
        
        for file_path, prob_id, step in files_by_iteration[iteration]:
            # Generate test file
            code_content = file_path.read_text(encoding='utf-8')
            json_content = jsonl_data[json_idx[prob_id]]
            step_id = json_content["sub_steps"][int(step) - 1]["step_number"]
            test_lst = json_content["sub_steps"][int(step) - 1]["test_cases"]
            
            test_file = tmp_dir / f'{step_id}_{iteration}.py'
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
                f.write("\nfrom scicode.parse.parse import process_hdf5_to_tuple\n")
                f.write(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)})\n")
                for idx in range(len(test_lst)):
                    f.write(f"target = targets[{idx}]\n\n")
                    for line in test_lst[idx].split('\n'):
                        f.write(line + '\n')
            
            # Run tests
            print(f'Testing function {step_id} (iteration {iteration})...')
            results['total_problems'][int(prob_id) - 1] += 1
            
            logs_dir_ = Path(log_dir, model_name, _get_background_dir(with_background))
            logs_dir_.mkdir(parents=True, exist_ok=True)
            logs_file = logs_dir_ / f'{step_id}_{iteration}.txt'
            
            if logs_file.exists():
                with open(logs_file, 'r') as f:
                    content = f.read().splitlines()
                    if content[0] == 'pass':
                        results['correct_problems'][int(prob_id) - 1] += 1
                        results['correct_steps'].add(step_id)
                        results['correct_dict'][prob_id].append(step_id)
                continue
            
            ret, output = run_script(test_file)
            if ret == 0:
                results['correct_problems'][int(prob_id) - 1] += 1
                results['correct_steps'].add(step_id)
                results['correct_dict'][prob_id].append(step_id)
                with open(logs_file, 'w') as f:
                    f.write('pass\n')
            else:
                results['failed_dict'][prob_id].append(step_id)
                with open(logs_file, 'w') as f:
                    f.write('fail\n' if ret == 1 else 'timeout\n')
                    f.write(output)
    
    # Calculate success rates for each iteration
    iteration_success = []
    for iteration in range(0, target_iteration + 1):
        results = iteration_results[iteration]
        total_steps = sum(results['total_problems'])
        correct_steps = len(results['correct_steps'])
        if total_steps > 0:
            success_rate = correct_steps / total_steps
            iteration_success.append(success_rate)
        else:
            iteration_success.append(0)
    
    # Plot success rate across iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, target_iteration + 1), iteration_success, marker='o')
    plt.title('Success Rate Across Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Success Rate')
    plt.grid(True)
    
    # Save plot
    plot_path = Path(output_dir) / f'{model_name}_iteration_success.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Save detailed results for target iteration
    target_results = iteration_results[target_iteration]
    correct_prob_num = sum(1 for i in range(PROB_NUM) 
                          if target_results['correct_problems'][i] == target_results['total_problems'][i]
                          and target_results['total_problems'][i] != 0)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    with open(output_dir / f'{model_name}_{_get_background_dir(with_background)}_summary.txt', 'w') as f:
        f.write(f'Results for iteration {target_iteration}:\n')
        f.write(f'Correct problems: {correct_prob_num}/{DEV_PROB_NUM if dev_set else PROB_NUM - DEV_PROB_NUM}\n')
        f.write(f'Correct steps: {len(target_results["correct_steps"])}/{DEV_STEP_NUM if dev_set else STEP_NUM}\n')
        f.write(f'Duration: {time.time() - start_time} seconds\n\n')
        f.write('Success rates across iterations:\n')
        for it, rate in enumerate(iteration_success, 1):
            f.write(f'Iteration {it}: {rate:.2%}\n')
    
    # Save detailed results
    results_data = {
        'iteration_success_rates': iteration_success,
        'target_iteration_results': {
            'correct_problems': target_results['correct_dict'],
            'failed_problems': target_results['failed_dict']
        }
    }
    
    with open(output_dir / f'{model_name}_{_get_background_dir(with_background)}_results.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    
    shutil.rmtree(tmp_dir)

def test_code(model_name, code_dir, log_dir, output_dir,
              jsonl_path, dev_set=False, with_background=False):

    jsonl_data = read_from_jsonl(jsonl_path)
    json_dct = {}
    json_idx = {}

    for prob_data in jsonl_data:
        json_dct[prob_data['problem_id']] = len(prob_data['sub_steps'])
        json_idx[prob_data['problem_id']] = jsonl_data.index(prob_data)
    start_time = time.time()

    code_dir_ = Path(code_dir, model_name, _get_background_dir(with_background))
    tmp_dir = Path(f'tmp_{start_time}')

    tmp_dir.mkdir(parents=True, exist_ok=True)

    for file_path in code_dir_.iterdir():
        if file_path.is_file():
            file_name = file_path.stem
            file_id = file_name.split(".")[0]
            file_step = file_name.split(".")[1]

            code_content = file_path.read_text(encoding='utf-8')
            json_content = jsonl_data[json_idx[file_id]]
            step_id = json_content["sub_steps"][int(file_step) - 1]["step_number"]
            test_lst = json_content["sub_steps"][int(file_step) - 1]["test_cases"]
            assert_file = Path(tmp_dir, f'{step_id}.py')
            with open(assert_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
                f.write(f"""

from scicode.parse.parse import process_hdf5_to_tuple

""")
                f.write(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)})" + '\n')
                for idx in range(len(test_lst)):
                    f.write(f"target = targets[{idx}]\n\n")
                    for line in test_lst[idx].split('\n'):
                        f.write(line + '\n')

    def run_script(script_path):
        try:
            subprocess.run(['python', script_path], check=True, capture_output=True,
                           text=True, timeout=180)
            return 0
        except subprocess.CalledProcessError as e:
            print(f"Error running script {script_path}: {e}")
            print(e.output)
            return 1
        except subprocess.TimeoutExpired as e:
            print(f"Runtime error while running script {script_path}: {e}")
            return 2

    correct_prob = np.zeros(PROB_NUM)
    tot_prob = np.zeros(PROB_NUM)
    correct_step = []
    correct_dict = {}

    for i in range(PROB_NUM):
        correct_dict[f'{i+1}'] = []

    num_steps_run = 0
    for file_path in tmp_dir.iterdir():
        if file_path.is_file():
            func_id = file_path.stem
            prob_id = func_id.split('.')[0]
            print(f'Testing function {func_id} ...')
            num_steps_run += 1
            tot_prob[int(prob_id) - 1] += 1
            logs_dir_ = Path(log_dir, model_name, _get_background_dir(with_background))
            logs_dir_.mkdir(parents=True, exist_ok=True)
            logs_file = Path(logs_dir_, f'{file_path.stem}.txt')
            if logs_file.exists():
                with open(logs_file, 'r') as f:
                    content = f.read().splitlines()
                    if content[0] == 'pass':
                        correct_prob[int(prob_id) - 1] += 1
                        correct_step.append(func_id)
                        correct_dict[prob_id].append(func_id)
                continue
            ret = run_script(file_path)
            if ret == 0:
                correct_prob[int(prob_id) - 1] += 1
                correct_step.append(func_id)
                correct_dict[str(prob_id)].append(func_id)
                with open(logs_file, 'w') as f:
                    f.write('pass')
            elif ret == 1:
                with open(logs_file, 'w') as f:
                    f.write('fail')
            else:
                with open(logs_file, 'w') as f:
                    f.write('time out')

    test_time = time.time() - start_time

    correct_prob_num = sum(1 for i in range(PROB_NUM) if
                           correct_prob[i] == tot_prob[i]
                           and tot_prob[i] != 0)

    print('number of steps actually run', num_steps_run)
    print(f'correct problems: {correct_prob_num}/{DEV_PROB_NUM if dev_set else PROB_NUM - DEV_PROB_NUM}')
    print(f'correct steps: {len(correct_step)}/{DEV_STEP_NUM if dev_set else STEP_NUM}')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{output_dir}/{model_name}_{_get_background_dir(with_background)}.txt', 'w') as f:
        f.write(f'correct problems: {correct_prob_num}/{DEV_PROB_NUM if dev_set else PROB_NUM - DEV_PROB_NUM}\n')
        f.write(f'correct steps: {len(correct_step)}/{DEV_STEP_NUM if dev_set else STEP_NUM}\n\n')
        f.write(f'duration: {test_time} seconds\n')
        f.write('\ncorrect problems: ')
        f.write(f'\n\n{[i + 1 for i in range(PROB_NUM) if correct_prob[i] == tot_prob[i] and tot_prob[i] != 0]}\n')

    with open(f'{output_dir}/{model_name}_{_get_background_dir(with_background)}.json', 'w', encoding='utf-8') as f:
        json.dump(correct_dict, f, indent=4)
    
    shutil.rmtree(tmp_dir)


def get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model name"
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=Path("eval_results", "generated_code"),
        help="Code directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Log directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results"),
        help="Eval results directory",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=Path("eval", "data", "problems_all.jsonl"),
        help="Path to jsonl file",
    )
    parser.add_argument(
        "--dev-set",
        action='store_true',
        help="Test dev set if enabled",
    ),
    parser.add_argument(
        "--with-background",
        action="store_true",
        help="Include problem background if enabled",
    )
    return parser


def main(model: str,
         code_dir: Path,
         log_dir: Path,
         output_dir: Path,
         jsonl_path: Path,
         dev_set: bool,
         with_background: bool
) -> None:
    if not Path(H5PY_FILE).exists():
        raise FileNotFoundError("Please download the numeric test results before testing generated code.")
    model = Path(model).parts[-1]
    #test_code(model, code_dir, log_dir, output_dir, jsonl_path, dev_set, with_background)
    #test_code_with_iterations(model, code_dir, log_dir, output_dir, jsonl_path, dev_set, with_background)
    test_code_with_iterations(model_name=model, code_dir=code_dir, log_dir=log_dir, output_dir=output_dir,
                              jsonl_path=jsonl_path, target_iteration=3, dev_set=dev_set, with_background=with_background)
if __name__ == "__main__":
    args = get_cli().parse_args()
    main(**vars(args))
