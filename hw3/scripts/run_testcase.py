import argparse

import os
import subprocess
import re
import cv2
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Release script for hw3.")
    parser.add_argument("id", type=int, help="The id of testcase to run.")
    parser.add_argument(
        "--cpu",
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
        "--debug", action="store_true", required=False, help="Enable debug mode."
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
        default="results",
        help="Directory to store output results.",
    )
    parser.add_argument(
        "--save_log",
        action="store_true",
        required=False,
        help="Save the output log to a file.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        required=False,
        help="Enable profiling mode.",
    )
    parser.add_argument(
        "--profile_visual",
        action="store_true",
        required=False,
        help="Output profile in visual format.",
    )
    parser.add_argument(
        "--profile_cpu",
        action="store_true",
        required=False,
        help="Enable CPU profiling mode.",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        required=False,
        help="Check kernel occupancy.",
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


def run_build(debug: bool, cpu: bool):
    build_type = "Debug" if debug else "Release"
    command = ["cmake", "--build", f"build/{build_type}", "-j"]
    if cpu:
        command.extend(["--target", "hw3_cpu"])
    subprocess.run(command, check=True)


def read_testcase(input_file: str) -> dict:
    data = {}
    with open(input_file, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            data[key] = value

    # Check for required keys
    required_keys = ["pos", "tarpos", "width", "height", "timelimit", "valid"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in testcase file.")

    pos_data = data["pos"].split(" ")
    if len(pos_data) != 3:
        raise ValueError("Position data must contain exactly three values.")
    data["pos"] = pos_data

    tarpos_data = data["tarpos"].split(" ")
    if len(tarpos_data) != 3:
        raise ValueError("Target position data must contain exactly three values.")
    data["tarpos"] = tarpos_data
    
    valid_path = data["valid"]
    valid_name = os.path.basename(valid_path)
    valid_path = os.path.join(os.path.dirname(input_file), valid_name)
    data["valid"] = valid_path

    return data


def make_output_path(output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        os.remove(path)
    return path


def run_testcase(
    id: int,
    data: dict,
    args: argparse.Namespace,
) -> dict:
    pos = data["pos"]
    tarpos = data["tarpos"]
    width = data["width"]
    height = data["height"]
    timelimit = data["timelimit"]
    valid = data["valid"]

    debug = getattr(args, "debug", False)
    cpu = getattr(args, "cpu", False)
    output_dir = getattr(args, "output_dir", "results")
    save_log = getattr(args, "save_log", False)
    profile = getattr(args, "profile", False)
    profile_visual = getattr(args, "profile_visual", False)
    profile_cpu = getattr(args, "profile_cpu", False)
    ncu = getattr(args, "ncu", False)
    dry_run = getattr(args, "dry_run", False)
    no_srun = getattr(args, "no_srun", False)

    result = {
        "output_path": None,
        "log_path": None,
        "valid_path": valid,
        "success": False,
        "elapsed_time": None,
    }

    build_type = "Debug" if debug else "Release"

    executable_name = "hw3_cpu" if cpu else "hw3"
    executable_path = os.path.join("build", build_type, executable_name)

    os.makedirs(output_dir, exist_ok=True)
    output_path = make_output_path(output_dir, f"{id:02d}.png")
    log_path = make_output_path(output_dir, f"{id:02d}.log")

    ncu_path = make_output_path(output_dir, f"{id:02d}.ncu-rep")

    prof_path = make_output_path(output_dir, f"{id:02d}.prof")
    prof_visual_path = make_output_path(output_dir, f"{id:02d}.nvvp")
    prof_json_path = make_output_path(output_dir, f"{id:02d}.prof.json")

    command = [
        executable_path,
        pos[0],
        pos[1],
        pos[2],
        tarpos[0],
        tarpos[1],
        tarpos[2],
        width,
        height,
        output_path,
    ]
    if ncu:
        ncu_command = [
            "ncu",
            "--set",
            "full",
            "--target-processes",
            "all",
            "--log-file",
            ncu_path,
        ]
        command = ncu_command + command
    if profile and not ncu:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        profile_command = ["nvprof"]
        if profile_cpu:
            profile_command.extend(["--cpu-profiling", "on"])
        if profile_visual:
            profile_command.extend(
                ["--output-profile", prof_visual_path, "--profile-api-trace", "all"]
            )
        else:
            profile_command.extend(["--log-file", prof_path])
        command = profile_command + command
    if not cpu and not no_srun:
        srun_command = (
            f"srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t {timelimit}".split(" ")
        )
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

    result["output_path"] = output_path
    if save_log:
        result["log_path"] = log_path

    if os.path.exists(output_path):
        result["success"] = True

    if os.path.exists(prof_path) and profile_visual:
        with open(prof_json_path, "w") as convert_output:
            subprocess.run(
                ["python", "nvprof2json/nvprof2json.py", prof_visual_path],
                check=True,
                stdout=convert_output,
            )

    elapsed_log = re.search(r"Elapsed: *([0-9.]+) us", log)
    if elapsed_log:
        result["elapsed_time"] = int(elapsed_log.group(1))

    return result


def validate(output_path: str, valid_path: str) -> float:
    output = cv2.imread(output_path)
    valid = cv2.imread(valid_path)
    if output is None or valid is None:
        raise ValueError("Output or valid image could not be read.")

    output = output.astype("float32")
    valid = valid.astype("float32")

    diff = cv2.absdiff(output, valid).sum()
    err = diff / valid.sum()

    return err * 100.0


if __name__ == "__main__":
    args = parse_arguments()

    if not args.no_build:
        run_build(args.debug, args.cpu)

    testcase_file = os.path.join(args.input_dir, f"{args.id:02d}.txt")
    testcase_data = read_testcase(testcase_file)

    result = run_testcase(
        id=args.id,
        args=args,
        data=testcase_data,
    )

    if result["elapsed_time"] is not None:
        print(f"Elapsed time: {result['elapsed_time']} us")

    if result["success"]:
        error_percentage = validate(result["output_path"], result["valid_path"])
        print(f"Validation error: {error_percentage:.4f}%")
