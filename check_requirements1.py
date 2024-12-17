import importlib
import subprocess

def is_installed(package):
    # Try to import the package to check if it is installed
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False

def check_requirements(requirements_file):
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()

    # Clean up each line to remove any extra whitespace and version specifiers
    requirements = [req.strip().split("==")[0] for req in requirements]

    for requirement in requirements:
        if is_installed(requirement):
            print(f"{requirement} is already installed.")
        else:
            print(f"{requirement} is NOT installed.")

if __name__ == "__main__":
    # Path to the requirements.txt file
    requirements_file = "requirements.txt"
    
    check_requirements(requirements_file)
