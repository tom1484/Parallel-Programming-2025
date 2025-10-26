import argparse

import os
import subprocess
import re
import cv2
import numpy as np

from run_testcase import run_build, read_testcase, run_testcase, validate


def parse_arguments():
    parser = argparse.ArgumentParser(description="Release script for hw3.")
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="Starting ID of testcases to run.",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=99,
        help="Ending ID of testcases to run.",
    )
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if not args.no_build:
        run_build(args.debug, args.cpu)

    testcase_list = []
    for file in os.listdir(args.input_dir):
        if file[-4:] != ".txt":
            continue
        id = int(file[:-4])
        if args.start_id <= id <= args.end_id:
            testcase_list.append(id)

    testcase_list.sort()
    
    print(f"Running {len(testcase_list)} testcases...")

    print("=======================================")
    print("| ID | Time (us) | Error (%) | Passed |")
    print("---------------------------------------")

    for id in testcase_list:
        testcase_file = os.path.join(args.input_dir, f"{id:02d}.txt")
        testcase_data = read_testcase(testcase_file)

        result = run_testcase(
            id=id,
            debug=args.debug,
            cpu=args.cpu,
            output_dir=args.output_dir,
            save_log=True,
            profile=False,
            data=testcase_data,
        )

        print(f"| {id:02d} ", end="")

        if result["elapsed_time"] is not None:
            print(f"| {result['elapsed_time']:9d} ", end="")
        else:
            print(f"|       N/A ", end="")

        error_percentage = 100.0
        if result["success"]:
            error_percentage = validate(result["output_path"], result["valid_path"])
            print(f"| {error_percentage:9.4f} ", end="")
        else:
            print(f"|       N/A ", end="")

        if error_percentage < 3.0:
            print("|    Yes |")
        else:
            print("|     No |")

    print("=======================================")
