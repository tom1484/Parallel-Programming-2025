import argparse

import os
import numpy as np

from run_testcase import (
    run_build,
    run_testcase,
    validate,
    make_output_path,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Release script for hw5.")
    parser.add_argument(
        "--testcases",
        type=list[int],
        default=[20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 512, 1024],
        help="List of testcase IDs to run.",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=10,
        help="Number of iterations for each testcase.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        required=False,
        help="Use CPU version of the program.",
    )
    parser.add_argument(
        "--no_build",
        action="store_true",
        required=False,
        help="Skip building the project.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="assets/testcases",
        help="Directory containing input testcases.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to store output results.",
    )
    parser.add_argument(
        "--no_srun",
        action="store_true",
        required=False,
        help="Do not use srun even for GPU execution.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.save_log = True

    if not args.no_build:
        run_build(args.sample)

    testcase_list = []
    for id in args.testcases:
        input_file = os.path.join(args.input_dir, f"b{id}.in")
        if os.path.exists(input_file):
            testcase_list.append(id)

    testcase_list.sort()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = make_output_path(args.output_dir, "benchmark.txt")
    with open(output_path, "w") as f:
        f.write("Benchmark\n\n")

    print(f"Running {len(testcase_list)} testcases...")

    print("================================")
    print("| ID | Time (us) | Valid Iters |")
    print("--------------------------------")

    for id in testcase_list:
        results = []
        elapsed_times = []
        is_valids = []
        all_passed = True

        with open(output_path, "a") as f:
            f.write(f"Testcase ID: {id:02d}\n")

        for i in range(args.iter):
            result = run_testcase(id=id, args=args)
            if result["elapsed_time"] is not None and result["success"]:
                elapsed_times.append(result["elapsed_time"])
                is_valid = validate(
                    result["output_path"], result["answer_path"], result["valid_path"]
                )
                is_valids.append(is_valid)
            else:
                all_passed = False

            results.append(result)

            with open(output_path, "a") as f:
                f.write(f"  Iteration {i+1}:\n")
                f.write(f"    Elapsed Time: {result['elapsed_time']} us\n")
                f.write(f"    Success: {result['success']}\n")
                if result["elapsed_time"] is not None and result["success"]:
                    is_valid = validate(
                        result["output_path"],
                        result["answer_path"],
                        result["valid_path"],
                    )
                    f.write(f"    Is Valid: {'Yes' if is_valid else 'No'}\n")

        all_passed = all_passed and all(is_valids)

        print(f"| {id:02d} ", end="")

        if len(elapsed_times) > 0:
            mean_elapsed_time = np.mean(elapsed_times).astype(int).item()
            print(f"| {mean_elapsed_time:9d} ", end="")
        else:
            print(f"|       N/A ", end="")

        valid_iters_str = f"{sum(list(map(int, is_valids)))}/{args.iter}"
        print(f"| {valid_iters_str:>11s} |")

        with open(output_path, "a") as f:
            f.write(f"  Summary:\n")
            if len(elapsed_times) > 0:
                f.write(f"    Mean Elapsed Time: {mean_elapsed_time} us\n")
            else:
                f.write(f"    Mean Elapsed Time: N/A\n")
            f.write(f"    All Passed: {'Yes' if all_passed else 'No'}\n")
            f.write("\n")

    print("============================================")
