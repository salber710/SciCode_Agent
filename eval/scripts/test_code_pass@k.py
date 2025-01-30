from pathlib import Path
import json
import subprocess
import time
import shutil
import numpy as np
import argparse
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import sys

from scicode.parse.parse import H5PY_FILE
from scicode.parse.parse import read_from_jsonl

PROB_NUM = 80
DEV_PROB_NUM = 15
STEP_NUM = 288
DEV_STEP_NUM = 50

def _get_background_dir(with_background):
    return "with_background" if with_background else "without_background"

def run_script(script_path):
    try:
        # Add current environment variables and Python path
        env = os.environ.copy()
        result = subprocess.run(
            ['python', script_path], 
            check=False,  # Don't raise exception on non-zero exit
            capture_output=True,
            text=True, 
            timeout=180,
            env=env
        )
        
        if result.returncode != 0:
            #print(f"\nError in {script_path}:")
            #print("STDOUT:", result.stdout)
            #print("STDERR:", result.stderr)
            return 1
            
        return 0
        
    except subprocess.TimeoutExpired as e:
        print(f"\nTimeout running {script_path}: {e}")
        return 2
    except Exception as e:
        print(f"\nUnexpected error running {script_path}: {str(e)}")
        return 1


from multiprocessing import Pool, cpu_count
from functools import partial

def test_single_variant(tmp_dir, jsonl_data, json_idx, code_dir_, variant_info):
    """Helper function to test a single variant."""
    substep_key, variant = variant_info
    problem_id, step_num = substep_key.split(".")
    file_name = f"{problem_id}.{step_num}_{variant}"
    source_file = Path(code_dir_, f"{file_name}.py")
    
    if not source_file.exists():
        return substep_key, variant, False
        
    # Create and run test file
    code_content = source_file.read_text(encoding='utf-8')
    json_content = jsonl_data[json_idx[problem_id]]
    step_id = json_content["sub_steps"][int(step_num) - 1]["step_number"]
    test_lst = json_content["sub_steps"][int(step_num) - 1]["test_cases"]
    
    assert_file = Path(tmp_dir, f'{file_name}_test.py')
    with open(assert_file, 'w', encoding='utf-8') as f:
        f.write("from scicode.parse.parse import process_hdf5_to_tuple\n")
        f.write("import numpy as np\n\n")
        f.write(code_content)
        f.write("\n\ntry:\n")
        f.write(f"    targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)})\n")
        for idx in range(len(test_lst)):
            f.write(f"    target = targets[{idx}]\n")
            for line in test_lst[idx].split('\n'):
                if line.strip():
                    f.write("    " + line + "\n")
            f.write("\n")
        f.write("except Exception as e:\n")
        f.write("    print(f'Error during execution: {str(e)}')\n")
        f.write("    raise e")
    
    ret = run_script(assert_file)
    return substep_key, variant, ret == 0

def analyze_sampling_effects(model_name, code_dir, log_dir, output_dir,
                           jsonl_path, dev_set=False, with_background=False):
    output_dir = Path(output_dir)
    suffix = "back" if with_background else "noBack"
    results_file = output_dir / f'{model_name}_pass@k_{suffix}.json'
    plot_file = output_dir / f'{model_name}_pass@k_{suffix}.png'

    N_values = [1, 3, 5, 7, 10]
    
    # Check if results already exist
    if results_file.exists():
        print(f"Loading existing results from {results_file}")
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        possible_variants = [i for i in range(10)]
        for N in N_values:
            variants_take = possible_variants[:N]
            num_pass, num_total = 0, 0
            for subproblem in saved_results['variant_results']:
                passed = False
                for variant in variants_take:
                    v_num, res = saved_results['variant_results'][subproblem][variant]
                    if N == 1:
                        print(v_num)
                    assert v_num == str(variant)
                    if res == 1:
                        passed = True
                num_total += 1
                if passed:
                    num_pass += 1
            print(f"For N={N}: {round(num_pass/num_total, 6) * 100}")

        
        return (
            saved_results['N_values'],
            saved_results['pass_rates'],
            {k: [(v[0], bool(v[1])) for v in vs] for k, vs in saved_results['variant_results'].items()}
        )
    
    N_values = [1, 3, 5, 7, 10]
    pass_rates = []
    
    # Read initial data
    jsonl_data = read_from_jsonl(jsonl_path)
    json_dct = {}
    json_idx = {}
    substep_variants = defaultdict(list)

    for prob_data in jsonl_data:
        json_dct[prob_data['problem_id']] = len(prob_data['sub_steps'])
        json_idx[prob_data['problem_id']] = jsonl_data.index(prob_data)

    code_dir_ = Path(code_dir, model_name, _get_background_dir(with_background))
    
    # Collect all variants
    print("Collecting variants...")
    for file_path in code_dir_.iterdir():
        if file_path.is_file():
            file_name = file_path.stem
            problem_id, step_num_variant = file_name.split(".")
            step_num, variant = step_num_variant.split("_")
            substep_key = f"{problem_id}.{step_num}"
            substep_variants[substep_key].append(variant)
    
    # Prepare for parallel processing
    start_time = time.time()
    tmp_dir = Path(f'tmp_{start_time}')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create list of all variant tests to run
    variant_tests = []
    for substep_key, variants in substep_variants.items():
        for variant in sorted(variants):
            variant_tests.append((substep_key, variant))
    
    print(f"Testing {len(variant_tests)} variants in parallel...")
    
    # Initialize multiprocessing pool
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    pool = Pool(processes=num_processes)
    
    # Create partial function with fixed arguments
    test_func = partial(test_single_variant, tmp_dir, jsonl_data, json_idx, code_dir_)
    
    # Run tests in parallel
    results = pool.map(test_func, variant_tests)
    pool.close()
    pool.join()
    
    # Process results
    variant_results = defaultdict(list)
    for substep_key, variant, passed in results:
        variant_results[substep_key].append((variant, passed))
    
    # Sort results for deterministic behavior
    for substep_key in variant_results:
        variant_results[substep_key].sort(key=lambda x: x[0])
    
    # Analyze different N values
    total_substeps = len(substep_variants)
    
    for N in N_values:
        print(f"Analyzing results for N={N}...")
        passed_substeps = 0
        
        for substep_key, results in variant_results.items():
            sample_size = min(N, len(results))
            sampled_results = results[:sample_size]
            
            if any(passed for _, passed in sampled_results):
                passed_substeps += 1
        
        pass_rate = passed_substeps / total_substeps if total_substeps else 0
        pass_rates.append(pass_rate)
        print(f"Pass rate for N={N}: {pass_rate:.2%}")
    
    # Cleanup
    shutil.rmtree(tmp_dir)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, pass_rates, 'bo-')
    plt.xlabel('Number of Samples (k)')
    plt.ylabel('pass@k')
    plt.title('pass@k vs k with Human-Derived Contexts')
    plt.grid(True)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(plot_file)
    
    # Save numerical results
    results = {
        'N_values': N_values,
        'pass_rates': pass_rates,
        'variant_results': {k: [(v[0], int(v[1])) for v in vs] for k, vs in variant_results.items()}
    }
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    return N_values, pass_rates, variant_results

def get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test code with N-sample analysis",
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
    )
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
    analyze_sampling_effects(model, code_dir, log_dir, output_dir, jsonl_path, dev_set, with_background)

if __name__ == "__main__":
    args = get_cli().parse_args()
    main(**vars(args))