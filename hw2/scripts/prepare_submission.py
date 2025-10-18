#!/usr/bin/env python3
"""
Submission preparation script for homework assignment.
This script:
1. Creates a submission/ folder
2. Merges headers and source code into submission/hw2.cpp
3. Extracts compiler flags from CMake and creates submission/Makefile
"""

import os
import re
import shutil
from pathlib import Path

SOURCES = [
    "hw2.cpp",
    "image.cpp",
    "sift.cpp",
]
HEADERS = [
    "stb/image_write.h",
    "stb/image.h",
    "image.hpp",
    "sift.hpp",
]


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


def extract_cmake_flags(build_dir="build/Release"):
    """Extract compiler flags from CMake-generated files."""
    flags_file = os.path.join(build_dir, "CMakeFiles/hw2.dir/flags.make")
    link_file = os.path.join(build_dir, "CMakeFiles/hw2.dir/link.txt")

    cxx_flags = ""
    cxx_defines = ""
    link_flags = ""

    # Read flags.make
    flags_content = read_file(flags_file)
    for line in flags_content.split("\n"):
        if line.startswith("CXX_FLAGS ="):
            cxx_flags = line.replace("CXX_FLAGS =", "").strip()
        elif line.startswith("CXX_DEFINES ="):
            cxx_defines = line.replace("CXX_DEFINES =", "").strip()

    # Read link.txt to extract libraries
    link_content = read_file(link_file)
    if link_content:
        # Extract library flags from link command
        if "libgomp.so" in link_content or "-fopenmp" in cxx_flags:
            link_flags += " -fopenmp"
        if "libpthread" in link_content or "pthread" in link_content:
            link_flags += " -lpthread"

    return {
        "cxx_flags": cxx_flags,
        "cxx_defines": cxx_defines,
        "link_flags": link_flags.strip(),
    }


def flatten_local_includes(content, local_headers, flattened_headers):
    """Flatten path of #include for local header files."""
    lines = content.split("\n")
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        # Check if this is a local include
        if stripped.startswith('#include "'):
            # Extract the included filename
            if match := re.match(r'#include\s+"([^"]+)"', stripped):
                if included_file := match.group(1):
                    # Check if it's one of our local headers
                    if included_file in local_headers:
                        header_idx = local_headers.index(included_file)
                        header_name = flattened_headers[header_idx]
                        # Replace with flattened include
                        line = f'#include "{header_name}"'

        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def copy_source(output_dir):
    """Copy and flatten source files into output directory."""
    # Ensure source and header files exist
    for src in SOURCES:
        if not os.path.exists(os.path.join("src", src)):
            raise FileNotFoundError(f"Source file src/{src} not found")
    for hdr in HEADERS:
        if not os.path.exists(os.path.join("include", hdr)):
            raise FileNotFoundError(f"Header file include/{hdr} not found")

    flattened_sources = [src.replace("/", "_") for src in SOURCES]
    flattened_headers = [hdr.replace("/", "_") for hdr in HEADERS]

    for header, flattened_header in zip(HEADERS, flattened_headers):
        content = read_file(os.path.join("include", header))
        if content:
            content = flatten_local_includes(content, HEADERS, flattened_headers)
            write_file(os.path.join(output_dir, flattened_header), content)

    for source, flattened_source in zip(SOURCES, flattened_sources):
        content = read_file(os.path.join("src", source))
        if content:
            content = flatten_local_includes(content, HEADERS, flattened_headers)
            write_file(os.path.join(output_dir, flattened_source), content)
    
    return flattened_sources, flattened_headers


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

    # Copy source files
    sources, headers = copy_source(submission_dir)
    print(f"Copied source files to {submission_dir}/")

    # Extract compiler flags
    print("Extracting compiler flags from CMake...")
    cmake_flags = extract_cmake_flags()
    print(f"Extracted flags: {cmake_flags}")

    # Create Makefile
    print("Creating Makefile...")
    # Try to use Makefile.template if it exists, otherwise use built-in template
    makefile_template_src = "assets/Makefile.template"
    if os.path.exists(makefile_template_src):
        makefile_template = read_file(makefile_template_src)
        print("Using Makefile.template")
    else:
        raise FileNotFoundError("Makefile.template not found.")

    makefile_content = makefile_template.format(
        cxx_flags=cmake_flags["cxx_flags"],
        cxx_defines=cmake_flags["cxx_defines"],
        link_flags=cmake_flags["link_flags"],
        source_files=" ".join(sources)
    )

    makefile_output = os.path.join(submission_dir, "Makefile")
    write_file(makefile_output, makefile_content)
    print(f"Created {makefile_output}")

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
    print("Note: Release configuration is used by default")


if __name__ == "__main__":
    main()
