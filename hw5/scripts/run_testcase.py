import argparse

import os
import subprocess
import re
import cv2
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Release script for hw5.")
    parser.add_argument("id", type=int, help="The id of testcase to run.")
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
        "--save_log",
        action="store_true",
        required=False,
        help="Save the output log to a file.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        required=False,
        help="Perform a dry run without executing the test case.",
    )
    parser.add_argument(
        "--no_srun",
        action="store_true",
        required=False,
        help="Do not use srun even for GPU execution.",
    )
    return parser.parse_args()


def run_build(sample: bool):
    command = ["cmake", "--build", f"build", "-j"]
    if sample:
        command.extend(["--target", "sample"])
    subprocess.run(command, check=True)


def make_output_path(output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        os.remove(path)
    return path


def run_testcase(
    id: int,
    args: argparse.Namespace,
) -> dict:
    sample = getattr(args, "sample", False)
    input_dir = getattr(args, "input_dir", "assets/testcases")
    output_dir = getattr(args, "output_dir", "outputs")
    save_log = getattr(args, "save_log", False)
    dry_run = getattr(args, "dry_run", False)
    no_srun = getattr(args, "no_srun", False) or sample

    result = {
        "input_path": None,
        "output_path": None,
        "valid_path": None,
        "log_path": None,
        "success": False,
        "elapsed_time": None,
    }

    executable_name = "sample" if sample else "hw5"
    executable_path = os.path.join("build", executable_name)
    input_path = os.path.join(input_dir, f"b{id}.in")
    answer_path = os.path.join(input_dir, f"b{id}.out")

    if sample:
        output_dir = os.path.join(output_dir, "sample")

    os.makedirs(output_dir, exist_ok=True)
    output_path = make_output_path(output_dir, f"b{id}.out")
    valid_path = make_output_path(output_dir, f"b{id}.valid")
    log_path = make_output_path(output_dir, f"b{id}.log")

    command = [
        executable_path,
        input_path,
        output_path,
    ]
    if not no_srun:
        srun_command = f"srun -t 00:10:00 --gres=gpu:2".split(" ")
        command = srun_command + command

    if dry_run:
        print("Dry run command:")
        print(" ".join(command))
        return result

    log = ""
    if save_log:
        with open(log_path, "w") as log_file:
            proc_result = subprocess.run(
                command, stdout=log_file, stderr=subprocess.STDOUT
            )
        with open(log_path, "r") as log_file:
            log = log_file.read()
    else:
        proc_result = subprocess.run(command, check=True)
        log = proc_result.stdout.decode("utf-8") if proc_result.stdout else ""

    if proc_result.returncode != 0:
        return result

    result["input_path"] = input_path
    result["answer_path"] = answer_path
    result["output_path"] = output_path
    result["valid_path"] = valid_path
    if save_log:
        result["log_path"] = log_path

    if os.path.exists(output_path):
        result["success"] = True

    elapsed_log = re.search(r"Elapsed: *([0-9.]+) us", log)
    if elapsed_log:
        result["elapsed_time"] = int(elapsed_log.group(1))

    return result


def validate(input_path: str, output_path: str, valid_path: str) -> bool:
    command = [
        "python",
        "scripts/validate.py",
        input_path,
        output_path,
    ]
    with open(valid_path, "w") as valid_file:
        proc_result = subprocess.run(
            command, stdout=valid_file, stderr=subprocess.STDOUT
        )
    if proc_result.returncode != 0:
        return False

    return True


if __name__ == "__main__":
    args = parse_arguments()

    if not args.no_build:
        run_build(args.sample)

    result = run_testcase(
        id=args.id,
        args=args,
    )

    if result["elapsed_time"] is not None:
        print(f"Elapsed time: {result['elapsed_time']} us")

    if result["success"]:
        is_valid = validate(
            result["output_path"], result["answer_path"], result["valid_path"]
        )
        print(f"Validation result: {'Yes' if is_valid else 'No'}")
