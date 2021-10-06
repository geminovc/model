
import subprocess, getpass, json
import sys

def get_gpu_usage():
    """
    Returns a dict which contains information about memory usage for each GPU.
    In the following output, the GPU with id "0" uses 5774 MB of 16280 MB.
    253 MB are used by other users, which means that we are using 5774 - 253 MB.
    {
        "0": {
            "used": 5774,
            "used_by_others": 253,
            "total": 16280
        },
        "1": {
            "used": 5648,
            "used_by_others": 253,
            "total": 16280
        }
    }
    """

    # Name of current user, e.g. "root"
    current_user = getpass.getuser()

    # Find mapping from process ids to user names
    command = ["ps", "axo", "pid,user"]
    output = subprocess.check_output(command).decode("utf-8")
    pid_user = dict(row.strip().split()
        for row in output.strip().split("\n")[1:])

    # Find all GPUs and their total memory
    command = ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv"]
    output = subprocess.check_output(command).decode("utf-8")
    total_memory = dict(row.replace(",", " ").split()[:2]
        for row in output.strip().split("\n")[1:])

    # Store GPU usage information for each GPU
    gpu_usage = {gpu_id: {"used": 0, "used_by_others": 0, "total": int(total)}
        for gpu_id, total in total_memory.items()}

    # Use nvidia-smi to get GPU memory usage of each process
    command = ["nvidia-smi", "pmon", "-s", "m", "-c", "1"]
    output = subprocess.check_output(command).decode("utf-8")
    for row in output.strip().split("\n"):
        if row.startswith("#"): continue

        gpu_id, pid, type, mb, command = row.split()

        # Special case to skip weird output when no process is running on GPU
        if pid == "-": continue

        gpu_usage[gpu_id]["used"] += int(mb)

        # If the GPU user is different from us
        if pid_user[pid] != current_user:
            gpu_usage[gpu_id]["used_by_others"] += int(mb)

    return gpu_usage

def get_free_gpus(max_usage_by_all_mb=1000):
    """
    Returns the ids of GPUs which are occupied to less than 1 GB by other users.
    """
    
    return [ int(gpu_id) -1 for gpu_id, usage in get_gpu_usage().items()
        if (usage["used_by_others"] + usage["used"] <= max_usage_by_all_mb and gpu_id != '0') ]

if __name__ == "__main__":
    sys.exit(get_free_gpus(max_usage_by_all_mb=0))