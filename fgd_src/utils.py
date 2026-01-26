import json
import logging
import os
import random
import socket
import subprocess
from typing import Dict, List, Optional, Tuple

from job import Job

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'

def get_logger(name: str, level=logging.DEBUG):
    """Create a logger with standard formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
    logger.addHandler(ch)
    return logger


def get_ip_address() -> str:
    """Get the IP address of this machine"""
    try:
        # Get the IP address by connecting to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to localhost
        return "127.0.0.1"


def get_num_gpus() -> int:
    """Get the number of GPUs available on this machine"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        return len(result.stdout.strip().split('\n'))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0


def read_all_throughputs_json_v2(throughputs_file: str) -> Dict:
    """
    Read throughput measurements from JSON file
    
    Format:
    {
        "worker_type": {
            ("model_name", scale_factor): {
                "null": throughput_value,
                ("other_model", scale_factor): [throughput1, throughput2]
            }
        }
    }
    """
    with open(throughputs_file, 'r') as f:
        data = json.load(f)
    
    # Convert string keys back to tuples
    throughputs = {}
    for worker_type, models in data.items():
        throughputs[worker_type] = {}
        for model_key_str, values in models.items():
            # Parse model key
            model_key = eval(model_key_str)  # Convert string to tuple
            throughputs[worker_type][model_key] = {}
            
            for other_key_str, throughput in values.items():
                if other_key_str == 'null':
                    throughputs[worker_type][model_key]['null'] = throughput
                else:
                    other_key = eval(other_key_str)
                    throughputs[worker_type][model_key][other_key] = throughput
    
    return throughputs


# def read_per_instance_type_spot_prices_json(prices_dir: str) -> Dict:
#     """Read spot instance prices from directory of JSON files"""
#     prices = {}
    
#     if not os.path.isdir(prices_dir):