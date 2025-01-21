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
            print(f"\nError in {script_path}:")
            #print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return 1
            
        return 0
        
    except subprocess.TimeoutExpired as e:
        print(f"\nTimeout running {script_path}: {e}")
        return 2
    except Exception as e:
        print(f"\nUnexpected error running {script_path}: {str(e)}")
        return 1

def test_code_with_sampling(model_name, code_dir, log_dir, output_dir,
                          jsonl_path, N, dev_set=False, with_background=False):
    
    jsonl_data = read_from_jsonl(jsonl_path)
    json_dct = {}
    json_idx = {}
    substep_variants = defaultdict(list)  # Track all variants of each substep

    for prob_data in jsonl_data:
        json_dct[prob_data['problem_id']] = len(prob_data['sub_steps'])
        json_idx[prob_data['problem_id']] = jsonl_data.index(prob_data)

    code_dir_ = Path(code_dir, model_name, _get_background_dir(with_background))
    
    # First pass: collect all variants for each substep
    for file_path in code_dir_.iterdir():
        if file_path.is_file():
            file_name = file_path.stem
            problem_id, step_num_variant = file_name.split(".")
            step_num, variant = step_num_variant.split("_")
            substep_key = f"{problem_id}.{step_num}"
            substep_variants[substep_key].append(variant)

    # Second pass: test sampled variants
    start_time = time.time()
    tmp_dir = Path(f'tmp_{start_time}')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    passed_substeps = set()
    total_substeps = set()

    def numeric_sort_key(item):
        problem_id, step_num = item[0].split('.')
        return (int(problem_id), int(step_num))
        
    # Sort substeps numerically
    for substep_key, variants in sorted(substep_variants.items(), key=numeric_sort_key):
        problem_id, step_num = substep_key.split(".")
        total_substeps.add(substep_key)
        
        # Sample N variants (or all if less than N available)
        sample_size = min(N, len(variants))
        sampled_variants = random.sample(variants, sample_size)
        
        passed_any = False
        for variant in sampled_variants:
            file_name = f"{problem_id}.{step_num}_{variant}"
            source_file = Path(code_dir_, f"{file_name}.py")
            
            if not source_file.exists():
                continue
                
            # Create test file
            code_content = source_file.read_text(encoding='utf-8')
            json_content = jsonl_data[json_idx[problem_id]]
            step_id = json_content["sub_steps"][int(step_num) - 1]["step_number"]
            test_lst = json_content["sub_steps"][int(step_num) - 1]["test_cases"]
            
            assert_file = Path(tmp_dir, f'{file_name}_test.py')
            with open(assert_file, 'w', encoding='utf-8') as f:
                # Write imports first (outside try block)
                f.write("from scicode.parse.parse import process_hdf5_to_tuple\n")
                f.write("import numpy as np\n\n")
                
                # Write the original code
                f.write(code_content)
                f.write("\n\n")
                
                # Start try block after the function definition
                f.write("try:\n")
                # Write test setup and assertions
                f.write(f"    targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)})\n")
                for idx in range(len(test_lst)):
                    f.write(f"    target = targets[{idx}]\n")
                    for line in test_lst[idx].split('\n'):
                        if line.strip():  # Only indent non-empty lines
                            f.write("    " + line + "\n")
                    f.write("\n")

                # Add except block
                f.write("except Exception as e:\n")
                f.write("    print(f'Error during execution: {str(e)}')\n")
                f.write("    raise e")
            
            # Run test
            ret = run_script(assert_file)
            if ret == 0:
                print("PASSED")
                passed_any = True
                break
        
        if passed_any:
            passed_substeps.add(substep_key)
    
    shutil.rmtree(tmp_dir)
    
    pass_rate = len(passed_substeps) / len(total_substeps) if total_substeps else 0
    return pass_rate

def analyze_sampling_effects(model_name, code_dir, log_dir, output_dir,
                           jsonl_path, dev_set=False, with_background=False):
    N_values = [1, 3, 5, 7]
    pass_rates = []
    
    for N in N_values:
        print(f"Testing with N={N} samples...")
        pass_rate = test_code_with_sampling(
            model_name, code_dir, log_dir, output_dir,
            jsonl_path, N, dev_set, with_background
        )
        pass_rates.append(pass_rate)
        print(f"Pass rate for N={N}: {pass_rate:.2%}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, pass_rates, 'bo-')
    plt.xlabel('Number of Samples (N)')
    plt.ylabel('Pass Rate')
    plt.title('Pass Rate vs Number of Samples')
    plt.grid(True)
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / f'{model_name}_sampling_analysis.png')
    
    # Save numerical results
    results = {
        'N_values': N_values,
        'pass_rates': pass_rates
    }
    with open(output_dir / f'{model_name}_sampling_results.json', 'w') as f:
        json.dump(results, f, indent=4)

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