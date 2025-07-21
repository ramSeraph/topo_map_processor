import os
import sys
import shutil
import subprocess
import importlib.resources as resources

def execute_bash_script(script_name):
    """
    Executes a bash script from the bash_scripts directory, locating it
    relative to the project structure.
    """
    if not shutil.which("bash"):
        raise FileNotFoundError("bash executable not found in PATH")

    # Use importlib.resources to access the script as package data
    # The script is located in topo_map_processor/bash_scripts
    try:
        script_path = resources.files('topo_map_processor.bash_scripts').joinpath(script_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"The script {script_name} does not exist or is not a file. Searched in package data.")

    command_name = sys.argv[0]
    os.environ["COMMAND_NAME"] = command_name
    # Execute the bash script and stream output, merging stderr into stdout
    process = subprocess.Popen(
        [shutil.which("bash"), str(script_path)] + sys.argv[1:],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        text=True,
        bufsize=1  # Line-buffered
    )

    # Stream combined stdout and stderr
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    process.wait()

    if process.returncode != 0:
        raise Exception(f"Bash script '{script_name}' failed with exit code {process.returncode}")

def generate_lists():
    """
    Executes the generate_lists.sh bash script.
    """
    execute_bash_script("generate_lists.sh")

def upload_to_release():
    """
    Executes the upload_to_release.sh bash script.
    """
    execute_bash_script("upload_to_release.sh")


if __name__ == "__main__":
    generate_lists()
