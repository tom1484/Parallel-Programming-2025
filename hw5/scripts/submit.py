#!/usr/bin/env python3
"""
Submission preparation script for homework assignment.
This script:
1. Creates a submission/ folder
2. Merges headers and source code into submission/hw5.cpp
3. Extracts compiler flags from CMake and creates submission/Makefile
"""

import os
import re
import shutil
from pathlib import Path


def read_file(filepath):
    """Read file content safely."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found")
        return ""


def write_file(filepath, content):
    """Write file content safely."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def remove_local_includes(content, local_headers):
    """Remove #include directives for local header files."""
    lines = content.split("\n")
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        # Check if this is a local include
        is_local_include = False
        if stripped.startswith('#include "'):
            # Extract the included filename
            match = re.match(r'#include\s+"([^"]+)"', stripped)
            if match:
                included_file = match.group(1)
                # Check if it's one of our local headers
                if any(header.endswith(included_file) for header in local_headers):
                    is_local_include = True

        if not is_local_include:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def sort_headers_by_dependency(header_contents):
    """Sort headers based on dependencies to ensure correct order."""
    from collections import defaultdict, deque

    # Build dependency graph
    graph = defaultdict(set)
    in_degree = defaultdict(int)

    for header, content in header_contents.items():
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#include "'):
                match = re.match(r'#include\s+"([^"]+)"', stripped)
                if match:
                    included_file = match.group(1)
                    # Find the full header name
                    for h in header_contents.keys():
                        if h.endswith(included_file):
                            graph[h].add(header)
                            in_degree[header] += 1
                            break

    # Topological sort using Kahn's algorithm
    zero_in_degree = deque([h for h in header_contents.keys() if in_degree[h] == 0])
    sorted_headers = []

    while zero_in_degree:
        current = zero_in_degree.popleft()
        sorted_headers.append(current)
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree.append(neighbor)

    if len(sorted_headers) != len(header_contents):
        raise ValueError("Cyclic dependency detected among headers")

    return sorted_headers


def merge_files():
    """Merge all header and source files into a single C++ file."""
    # Define file order based on dependencies
    headers = [
    ]

    sources = [
        "main.cpp",
    ]

    merged_content = []

    # Add header comment
    merged_content.append("// Merged submission file for homework assignment")
    merged_content.append("// Generated automatically - do not edit manually")
    merged_content.append("")

    # Collect all local header filenames for filtering
    local_headers = headers.copy()

    # Process headers first
    merged_content.append("// ========== HEADER FILES ==========")
    merged_content.append("")

    header_contents = {}
    for header in headers:
        header_file_path = f"include/{header}"
        if os.path.exists(header_file_path):
            content = read_file(header_file_path)
            header_contents[header] = content
        else:
            raise FileNotFoundError(f"Header file {header} not found.")

    header_orders = sort_headers_by_dependency(header_contents)
    for header in header_orders:
        print(f"Processing header: {header}")
        content = header_contents[header]
        # Remove include guards and local includes
        content = remove_local_includes(content, local_headers)

        merged_content.append(f"// From {header}")
        merged_content.append(content)
        merged_content.append("")

    # Process source files
    merged_content.append("// ========== SOURCE FILES ==========")
    merged_content.append("")

    for source_file in sources:
        source_file_path = f"src/{source_file}"
        if os.path.exists(source_file_path):
            print(f"Processing source: {source_file}")
            content = read_file(source_file_path)

            # Remove local includes
            content = remove_local_includes(content, local_headers)

            merged_content.append(f"// From {source_file}")
            merged_content.append(content)
            merged_content.append("")

    return "\n".join(merged_content)


def main():
    """Main function to prepare submission."""
    print("Preparing submission...")

    # Create submission directory
    submission_dir = "b10901002"
    if os.path.exists(submission_dir):
        print(f"Removing existing {submission_dir} directory...")
        for child in Path(submission_dir).iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    os.makedirs(submission_dir, exist_ok=True)
    print(f"Created {submission_dir} directory")

    # Merge source files
    print("Merging header and source files...")
    merged_cpp = merge_files()

    # Write merged C++ file
    cpp_output = os.path.join(submission_dir, "hw5.cpp")
    write_file(cpp_output, merged_cpp)
    print(f"Created {cpp_output}")

    # Copy Makefile
    makefile_src = "assets/Makefile"
    makefile_dst = os.path.join(submission_dir, "Makefile")
    if os.path.exists(makefile_src):
        shutil.copy(makefile_src, makefile_dst)
        print(f"Copied Makefile to {makefile_dst}")
    else:
        print("Warning: Makefile not found in assets/")

    # Copy report.pdf if exists
    report_src = "assets/report.pdf"
    report_dst = os.path.join(submission_dir, "report.pdf")
    if os.path.exists(report_src):
        shutil.copy(report_src, report_dst)
        print(f"Copied report to {report_dst}")
    else:
        print("Warning: report.pdf not found in assets/")

    print("\nSubmission preparation complete!")
    print(f"Files created in {submission_dir}/:")
    for file in os.listdir(submission_dir):
        file_path = os.path.join(submission_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file} ({size} bytes)")
        else:
            print(f"  {file}/ (directory)")

    print(f"\nTo build: cd {submission_dir} && make")
    print(f"To test: cd {submission_dir} && make test")
    print("Note: Release configuration is used by default")


if __name__ == "__main__":
    main()
