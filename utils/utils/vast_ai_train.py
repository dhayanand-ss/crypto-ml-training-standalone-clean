"""
Vast.ai Instance Management for Distributed ML Training

This module provides functions to create and manage Vast.ai GPU instances
for parallel machine learning model training. It integrates with Apache Airflow
for automated training pipelines.

Key Features:
- Automatic cost optimization (budget enforcement)
- Blacklist mechanism for problematic machines
- Automatic retry logic with timeouts
- Instance lifecycle management
"""

import os
import json
import time
import subprocess
import logging
import pickle
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Constants
BUDGET = 0.25  # $0.25 per hour maximum
MAX_POD_WAIT = 600  # 10 minutes maximum wait for pod to become "running"
MAX_RETRY_TIME = 1200  # 20 minutes maximum time to retry finding available pods
FIND_POD_SLEEP = 30  # 30 seconds between offer searches

# GPU Requirements
GPU_QUERY = "gpu_total_ram>=11 disk_space>=30 verified=True datacenter=True"

# Docker Image - configurable via environment variable
# Default to a public Python image (Vast AI instances have CUDA drivers pre-installed)
# The startup command clones repo and installs dependencies, so we just need a base image
DOCKER_IMAGE = os.getenv(
    "VASTAI_DOCKER_IMAGE",
    "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"  # Robust PyTorch image with CUDA support
)

# Blacklist storage path (configurable via environment variable)
BLACKLIST_DIR = os.getenv(
    "VASTAI_BLACKLIST_DIR",
    "/opt/airflow/custom_persistent_shared"
)
BLACKLIST_FILE = os.path.join(BLACKLIST_DIR, "blacklisted_machines.pkl")


def get_vastai_api_key() -> str:
    """Get Vast.ai API key from environment variable."""
    key = os.getenv("VASTAI_API_KEY")
    if not key:
        raise ValueError("VASTAI_API_KEY environment variable not set.")
    return key


def setup_vastai_cli():
    """Configure Vast.ai CLI with API key."""
    key = get_vastai_api_key()
    try:
        subprocess.run(
            ["vastai", "set", "api-key", key],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Vast.ai CLI configured successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to set Vast.ai API key: {e}")
        raise
    except FileNotFoundError:
        raise FileNotFoundError(
            "Vast.ai CLI not found. Install with: pip install vastai"
        )


def copy_data_to_instance(instance_id: str):
    """
    Directly copy required data files from host to Vast.ai instance.
    This bypasses the need for GCS credentials on the remote instance.
    """
    logger.info(f"Copying training data to instance {instance_id}...")


def get_ssh_info(instance_id: str) -> tuple[str, str, str]:
    """Parse SSH URL for an instance. Returns (host, port, user)."""
    try:
        cmd = ["vastai", "ssh-url", instance_id]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        url = result.stdout.strip()
        # format: ssh://root@host:port
        if "://" in url:
            url = url.split("://")[1]
        
        user_host, port = url.split(":")
        user, host = user_host.split("@")
        return host, port, user
    except Exception as e:
        logger.error(f"Failed to get SSH info for {instance_id}: {e}")
        raise

def get_ssh_identity_path() -> Optional[str]:
    """
    Find a valid SSH private key.
    Checks env var SSH_IDENTITY_FILE, then ~/.ssh/id_rsa, then ~/.ssh/lightning_rsa.
    """
    # 1. Check environment variable
    env_path = os.getenv("SSH_IDENTITY_FILE")
    if env_path and os.path.exists(env_path):
        return env_path
        
    # 2. Check standard paths in user's .ssh directory
    home = os.path.expanduser("~")
    ssh_dir = os.path.join(home, ".ssh")
    
    candidates = ["id_rsa", "lightning_rsa", "id_ed25519"]
    for name in candidates:
        key_path = os.path.join(ssh_dir, name)
        if os.path.exists(key_path):
            logger.info(f"Found SSH identity file: {key_path}")
            return key_path
            
    logger.warning("No standard SSH identity file found. SSH might fail if agent is not running.")
    return None

def run_ssh_command(instance_id: str, command: str):
    """Run command via SSH direct execution."""
    host, port, user = get_ssh_info(instance_id)
    key_path = get_ssh_identity_path()
    
    cmd = ["ssh", "-p", port, "-o", "StrictHostKeyChecking=no"]
    if key_path:
        cmd.extend(["-i", key_path])
        
    cmd.extend([f"{user}@{host}", command])
    
    # Retry logic for SSH connection (often fails immediately after boot)
    max_retries = 30
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                logger.warning(f"SSH command failed (attempt {attempt+1}/{max_retries}). Retrying in 10s...")
                time.sleep(10)
            else:
                logger.error(f"SSH command failed after {max_retries} attempts: {e}")
                # Log stderr for debugging authentication issues
                if e.stderr:
                    logger.error(f"SSH Error Output: {e.stderr.decode('utf-8')}")
                raise

def copy_file_scp(instance_id: str, local_path: str, remote_path: str):
    """Copy file via SCP."""
    host, port, user = get_ssh_info(instance_id)
    key_path = get_ssh_identity_path()
    
    cmd = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no"]
    if key_path:
        cmd.extend(["-i", key_path])
        
    cmd.extend([local_path, f"{user}@{host}:{remote_path}"])
    
    # Retry logic for SCP
    max_retries = 30
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                logger.warning(f"SCP command failed (attempt {attempt+1}/{max_retries}). Retrying in 10s...")
                time.sleep(10)
            else:
                logger.error(f"SCP command failed after {max_retries} attempts: {e}")
                raise


def copy_data_to_instance(instance_id: str):
    """
    Copy data files to the instance using SCP.
    """
    # Calculate the repository name (MUST match build_startup_command logic)
    github_repo = os.getenv("VASTAI_GITHUB_REPO", "")
    if github_repo:
        repo_name = github_repo.split("/")[-1].replace(".git", "")
    else:
        # Fallback to standard name used in build_startup_command Priority 3
        repo_name = "crypto-ml-training"
        
    # Define data files to copy (host path -> remote path)
    # Check for environmental overrides first
    prices_path_src = os.getenv("VASTAI_DATA_PRICES_PATH", "data/prices/BTCUSDT.csv")
    prices_test_src = os.getenv("VASTAI_DATA_PRICES_TEST_PATH", "data/prices/BTCUSDT_test.csv")
    articles_path_src = os.getenv("VASTAI_DATA_ARTICLES_PATH", "data/articles/articles.csv")

    data_files = {
        prices_path_src: f"/workspace/{repo_name}/data/prices/BTCUSDT.csv",
        prices_test_src: f"/workspace/{repo_name}/data/prices/BTCUSDT_test.csv",
        articles_path_src: f"/workspace/{repo_name}/data/articles/articles.csv"
    }

    # Add GCP credentials to file list if configured
    # INSERT AT BEGINNING to ensure it's uploaded first
    gcp_creds_src = os.getenv("GCP_CREDENTIALS_PATH")
    if gcp_creds_src and os.path.exists(gcp_creds_src):
        logger.info(f"Adding GCP credentials to upload list (PRIORITY): {gcp_creds_src}")
        # Create a new dict with credentials first
        new_data_files = {gcp_creds_src: "/workspace/gcp-credentials.json"}
        new_data_files.update(data_files)
        data_files = new_data_files
    else:
        logger.warning("GCP_CREDENTIALS_PATH not set or file not found. Skipping credentials upload.")
    
    for host_path, remote_path in data_files.items():
        if not os.path.exists(host_path):
            logger.warning(f"Local data file not found: {host_path}. Skipping upload.")
            continue
            
        try:
            # Ensure remote directory exists
            remote_dir = os.path.dirname(remote_path)
            logger.info(f"Ensuring remote directory exists: {remote_dir}")
            run_ssh_command(instance_id, f"mkdir -p {remote_dir}")
            
            # Copy file
            logger.info(f"Uploading {host_path}...")
            copy_file_scp(instance_id, host_path, remote_path)
            
        except Exception as e:
            logger.error(f"Failed to copy {host_path} to instance: {e}")


def load_blacklist() -> Set[int]:
    """Load blacklisted machine IDs from persistent storage."""
    if os.path.exists(BLACKLIST_FILE):
        try:
            with open(BLACKLIST_FILE, "rb") as f:
                blacklist = pickle.load(f)
                logger.info(f"Loaded {len(blacklist)} blacklisted machines")
                return set(blacklist)
        except Exception as e:
            logger.warning(f"Failed to load blacklist: {e}. Starting with empty blacklist.")
            return set()
    else:
        # Create directory if it doesn't exist
        os.makedirs(BLACKLIST_DIR, exist_ok=True)
        return set()


def save_blacklist(blacklist: Set[int]):
    """Save blacklisted machine IDs to persistent storage."""
    try:
        os.makedirs(BLACKLIST_DIR, exist_ok=True)
        with open(BLACKLIST_FILE, "wb") as f:
            pickle.dump(list(blacklist), f)
        logger.info(f"Saved {len(blacklist)} blacklisted machines to {BLACKLIST_FILE}")
    except Exception as e:
        logger.error(f"Failed to save blacklist: {e}")


def calculate_full_pod_cost(
    pod: dict,
    hours: float = 1,
    extra_storage_tb: float = 0,
    internet_up_tb: float = 0.005,  # 5 GB upload
    internet_down_tb: float = 0.01   # 10 GB download
) -> float:
    """
    Calculate total cost for a pod including storage and bandwidth.
    
    Args:
        pod: Pod offer dictionary from Vast.ai
        hours: Number of hours to run
        extra_storage_tb: Additional storage in TB beyond base
        internet_up_tb: Upload bandwidth in TB
        internet_down_tb: Download bandwidth in TB
    
    Returns:
        Total cost in USD
    """
    # Base hourly cost
    dph_total = pod.get("dph_total", 0)
    base_cost = dph_total * hours
    
    # Storage cost (prorated monthly)
    storage_monthly_cost = pod.get("storage_cost", 0)
    storage_cost = (storage_monthly_cost / (30 * 24)) * extra_storage_tb * hours
    
    # Internet costs
    internet_up_cost_per_tb = pod.get("inet_up_cost", 0)
    internet_down_cost_per_tb = pod.get("inet_down_cost", 0)
    internet_cost = (
        internet_up_cost_per_tb * internet_up_tb +
        internet_down_cost_per_tb * internet_down_tb
    )
    
    total_cost = base_cost + storage_cost + internet_cost
    return total_cost


def is_network_error(error_output: str) -> bool:
    """Check if error is a network/connection error."""
    network_error_keywords = [
        "ConnectionError",
        "Connection aborted",
        "Name or service not known",
        "Failed to establish",
        "Remote end closed connection",
        "Max retries exceeded",
        "socket.gaierror",
        "NewConnectionError"
    ]
    error_lower = error_output.lower()
    return any(keyword.lower() in error_lower for keyword in network_error_keywords)


def get_offers(max_retries: int = 3, retry_delay: int = 5) -> List[dict]:
    """
    Search for available GPU offers matching requirements.
    
    Args:
        max_retries: Maximum number of retry attempts for network errors
        retry_delay: Initial delay between retries (exponential backoff)
    
    Returns:
        List of pod offer dictionaries
    """
    cmd = [
        "vastai", "search", "offers",
        "--raw",
        GPU_QUERY
    ]
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            # Parse JSON output
            offers = json.loads(result.stdout)
            if not isinstance(offers, list):
                offers = [offers] if offers else []
            
            logger.info(f"Found {len(offers)} available offers")
            return offers
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or str(e)
            is_network = is_network_error(error_msg)
            
            if is_network and attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Network error searching offers (attempt {attempt + 1}/{max_retries}): {error_msg[:200]}"
                )
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Failed to search offers: {e}")
                if e.stderr:
                    logger.error(f"Error output: {e.stderr[:500]}")
                return []
                
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Timeout searching offers (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error("Timeout searching offers after all retries")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse offers JSON: {e}")
            return []
    
    return []


def verify_instance_exists(instance_id: str) -> Optional[Dict]:
    """
    Verify that an instance ID actually exists by listing all instances.
    
    Args:
        instance_id: Vast.ai instance ID to verify
    
    Returns:
        Instance dict if found, None otherwise
    """
    try:
        cmd = ["vastai", "show", "instances", "--raw"]
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        instances = json.loads(result.stdout)
        if not isinstance(instances, list):
            instances = [instances] if instances else []
        
        # Look for the instance ID
        for inst in instances:
            inst_id = str(inst.get("id", ""))
            if inst_id == str(instance_id):
                logger.info(f"Verified instance {instance_id} exists in instance list")
                return inst
        
        logger.warning(f"Instance {instance_id} not found in instance list")
        logger.debug(f"Available instance IDs: {[str(i.get('id', '')) for i in instances]}")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to verify instance existence: {e}")
        return None


def find_newly_created_instance(before_instances: List[Dict], after_instances: List[Dict]) -> Optional[str]:
    """
    Find the newly created instance by comparing before/after instance lists.
    
    Args:
        before_instances: List of instances before creation
        after_instances: List of instances after creation
    
    Returns:
        Instance ID of newly created instance, or None
    """
    before_ids = {str(inst.get("id", "")) for inst in before_instances}
    after_ids = {str(inst.get("id", "")) for inst in after_instances}
    
    new_ids = after_ids - before_ids
    if new_ids:
        new_id = list(new_ids)[0]
        logger.info(f"Found newly created instance: {new_id}")
        return new_id
    
    return None


def get_all_instances() -> List[Dict]:
    """Get all instances from Vast.ai."""
    try:
        cmd = ["vastai", "show", "instances", "--raw"]
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        instances = json.loads(result.stdout)
        if not isinstance(instances, list):
            instances = [instances] if instances else []
        
        return instances
    except Exception as e:
        logger.warning(f"Failed to get instances: {e}")
        return []


def wait_for_pod(instance_id: str, timeout: int = MAX_POD_WAIT) -> bool:
    """
    Wait for pod to reach "running" state.
    
    Args:
        instance_id: Vast.ai instance ID
        timeout: Maximum time to wait in seconds
    
    Returns:
        True if pod is running, False if timeout
    """
    start_time = time.time()
    logger.info(f"Waiting for instance {instance_id} to become running...")
    
    while time.time() - start_time < timeout:
        try:
            cmd = ["vastai", "show", "instance", instance_id, "--raw"]
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Parse JSON with error handling
            try:
                instance_data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse instance data JSON: {e}")
                logger.debug(f"Raw output: {result.stdout[:200]}")
                time.sleep(10)
                continue
            
            # Handle None case - get returns default only if key doesn't exist, not if value is None
            actual_status = instance_data.get("actual_status")
            if actual_status is None:
                logger.debug(f"Instance {instance_id} status not available yet, waiting...")
                time.sleep(10)
                continue
            
            status = str(actual_status).lower()
            
            if status == "running":
                logger.info(f"Instance {instance_id} is now running")
                return True
            elif status in ["failed", "stopped", "terminated"]:
                logger.error(f"Instance {instance_id} failed with status: {status}")
                return False
            
            logger.debug(f"Instance {instance_id} status: {status}, waiting...")
            time.sleep(10)  # Check every 10 seconds
            
        except subprocess.CalledProcessError as e:
            # Only log detailed error occasionally to avoid log spam
            elapsed = time.time() - start_time
            if int(elapsed) % 60 == 0:  # Log detailed error every minute
                logger.warning(f"Failed to check instance status: {e}")
                if e.stderr:
                    logger.warning(f"Error output: {e.stderr[:200]}")
            else:
                logger.debug(f"Failed to check instance status (will retry): {e}")
            time.sleep(10)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse instance data: {e}")
            time.sleep(10)
    
    logger.error(f"Timeout waiting for instance {instance_id} to become running")
    return False


def build_startup_command() -> str:
    """
    Build the startup command for the Vast.ai instance.
    
    This command will:
    1. Export environment variables
    2. Clone the repository
    3. Install Weights & Biases
    4. Start training
    
    Returns:
        Multi-line bash script as a single string
    """
    # Get environment variables
    env_vars = {
        "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL", ""),
        "MLFLOW_URI": os.getenv("MLFLOW_URI", ""),
        "MLFLOW_TRACKING_USERNAME": os.getenv("MLFLOW_TRACKING_USERNAME", ""),
        "MLFLOW_TRACKING_PASSWORD": os.getenv("MLFLOW_TRACKING_PASSWORD", ""),
        "MLFLOW_SQLALCHEMY_POOL_SIZE": os.getenv("MLFLOW_SQLALCHEMY_POOL_SIZE", "2"),
        "MLFLOW_SQLALCHEMY_MAX_OVERFLOW": os.getenv("MLFLOW_SQLALCHEMY_MAX_OVERFLOW", "0"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "auto"),
        "S3_URL": os.getenv("S3_URL", ""),
        "DATABASE_URL": os.getenv("DATABASE_URL", ""),
        "TRL_DATABASE_URL": os.getenv("TRL_DATABASE_URL", ""),
        "AIRFLOW_DB": os.getenv("AIRFLOW_DB", ""),
    }
    
    export_cmds_list = [
        f'export {key}="{value}"'
        for key, value in env_vars.items()
        if value  # Only export non-empty values
    ]
    # Add project root to PYTHONPATH explicitly
    export_cmds_list.append('export PYTHONPATH="/workspace/crypto-ml-training-standalone:$PYTHONPATH"')
    export_cmds = " && ".join(export_cmds_list) if export_cmds_list else ""
    
    # W&B API key (if available)
    wandb_key = os.getenv("WANDB_API_KEY", "")
    wandb_login = f"wandb login {wandb_key}" if wandb_key else "echo 'WANDB_API_KEY not set, skipping wandb login'"
    
    # Build complete startup command as a single line with semicolons
    # Vast.ai CLI works better with single-line commands or properly escaped multi-line
    cmd_parts = ["set -e"]
    
    # 0. Self-healing: Ensure basic tools are present (Vast.ai needs ssh/git)
    cmd_parts.append("apt-get update && apt-get install -y git openssh-client openssh-server rsync libgomp1 || echo 'Apt failed, continuing...'")
    
    if export_cmds:
        cmd_parts.append(export_cmds)
    
    # Check if using a custom Docker image (code pre-packaged) or need to clone/upload
    custom_image = os.getenv("VASTAI_DOCKER_IMAGE", "")
    if not custom_image:
        custom_image = DOCKER_IMAGE  # Use default if not set
    github_repo = os.getenv("VASTAI_GITHUB_REPO", "")
    
    # If using a custom image (not the default python:3.10-slim), code should be pre-packaged
    # Custom images typically have a registry prefix (e.g., docker.io/user/image or user/image)
    default_image = "python:3.10-slim"
    using_custom_image = custom_image != default_image and "/" in custom_image
    
    # Ensure workspace exists before we CD into it
    cmd_parts.append("mkdir -p /workspace")
    cmd_parts.append("cd /workspace")
    
    # Priority 1: Clone from GitHub if repository URL is provided
    if github_repo:
        repo_name = github_repo.split("/")[-1].replace(".git", "")
        # Use single string for the clone + cd logic to avoid '&&' syntax errors
        # FORCE FRESH CLONE: Delete existing directory to ensure no stale code
        cmd_parts.append(f"rm -rf {repo_name} || true")
        cmd_parts.append(f"git clone {github_repo} {repo_name}")
        cmd_parts.append(f"cd {repo_name}")
        cmd_parts.append("ls -R utils/trainer || echo 'Warning: Could not list utils/trainer'") # Debugging
        cmd_parts.extend([
            "pip install --upgrade pip",
            "pip install -r requirements.txt || echo 'Warning: requirements.txt not found, continuing...'",
        ])
    # Priority 2: Use code pre-packaged in Docker image
    elif using_custom_image:
        # Check standard locations: /workspace/crypto-ml-training-standalone or /app
        # Wrap the whole IF in a single string to avoid join issues
        cmd_parts.append(
            "if [ -d /workspace/crypto-ml-training-standalone ]; then cd /workspace/crypto-ml-training-standalone; "
            "elif [ -d /app ]; then cd /app; "
            "else echo 'Error: Code directory not found'; exit 1; fi"
        )
        logger.info(f"Using custom Docker image {custom_image} - code should be pre-packaged")
    # Priority 3: Expect manual upload
    else:
        cmd_parts.append("mkdir -p crypto-ml-training && cd crypto-ml-training")
        logger.warning("No GitHub repository or custom Docker image configured.")
        logger.warning("Code must be uploaded manually via SSH. Set VASTAI_GITHUB_REPO or build a custom Docker image.")
        cmd_parts.extend([
            "pip install --upgrade pip",
            "pip install -r requirements.txt || echo 'Warning: requirements.txt not found, continuing...'",
        ])
    
    # Common steps for all scenarios
    cmd_parts.extend([
        "pip install wandb || echo 'Warning: wandb installation failed'",
        wandb_login,
        # Pass Airflow context as environment variables if available
        f"export AIRFLOW_RUN_ID='{os.getenv('AIRFLOW_RUN_ID', '')}'",
        f"export AIRFLOW_DAG_ID='{os.getenv('AIRFLOW_DAG_ID', '')}'",
        f"export AIRFLOW_TASK_ID='{os.getenv('AIRFLOW_TASK_ID', '')}'",
        
        # Ensure data directories exist
        "mkdir -p data/prices data/articles",
        
        # Wait for data to be uploaded via copy_data_to_instance (avoids race condition)
        "echo 'Waiting for data upload...'",
    ])

    # Wait for GCP credentials IF configured AND FOUND locally
    gcp_creds_path = os.getenv("GCP_CREDENTIALS_PATH")
    if gcp_creds_path and os.path.exists(gcp_creds_path):
        cmd_parts.append("while [ ! -f /workspace/gcp-credentials.json ]; do echo 'Waiting for GCP credentials...'; sleep 5; done")
        
    cmd_parts.extend([
        "while [ ! -f data/prices/BTCUSDT.csv ] || [ ! -f data/articles/articles.csv ]; do sleep 5; done",
        "echo 'Data found. Starting training...'",
        
        # Run only LightGBM and TST training scripts (TRL disabled as per user request)
        # Scripts use relative paths and expect data in data/ folder
        "python -m utils.trainer.lgb_train --coin BTCUSDT || echo 'LightGBM training failed'",
        "python -m utils.trainer.tst_train --coin BTCUSDT || echo 'TST training failed'"
    ])
    
    # Add Google Cloud credentials setup if file exists
    if gcp_creds_path and os.path.exists(gcp_creds_path):
        # We'll upload this file separately via SCP/VastAI copy
        # Here we just ensure the env var points to the destination
        cmd_parts.insert(1, 'export GOOGLE_APPLICATION_CREDENTIALS="/workspace/gcp-credentials.json"')
    
    startup_cmd = " && ".join(cmd_parts)
    
    return startup_cmd



def create_instance(DEBUG: bool = False, **kwargs) -> Optional[str]:
    """
    Create a Vast.ai instance for distributed ML training.
    
    This function:
    1. Cleans up existing instances
    2. Searches for available GPU offers within budget
    3. Filters by blacklist
    4. Creates instance with Docker image and startup command
    5. Waits for instance to become ready
    6. Blacklists machine if instance fails to start
    
    Args:
        DEBUG: Enable debug logging
        **kwargs: Arbitrary keyword arguments (used for Airflow context)
    
    Returns:
        Instance ID if successful, None otherwise
    """
    # Log GCP credentials path if set
    gcp_creds_path = os.getenv("GCP_CREDENTIALS_PATH")
    if gcp_creds_path:
        logger.info(f"GCP_CREDENTIALS_PATH is set to: {gcp_creds_path}. Credentials will be uploaded.")
    else:
        logger.info("GCP_CREDENTIALS_PATH is not set. Skipping GCP credentials upload.")

    if "ti" in kwargs:
        # We are running in Airflow context
        ti = kwargs["ti"]
        os.environ["AIRFLOW_RUN_ID"] = ti.run_id
        os.environ["AIRFLOW_DAG_ID"] = ti.dag_id
        os.environ["AIRFLOW_TASK_ID"] = ti.task_id
        logger.info(f"Running in Airflow context: DAG={ti.dag_id}, Task={ti.task_id}, Run={ti.run_id}")

    if DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("Starting Vast.ai Instance Creation")
    logger.info("=" * 60)
    
    # Setup Vast.ai CLI
    try:
        setup_vastai_cli()
    except Exception as e:
        logger.error(f"Failed to setup Vast.ai CLI: {e}")
        return None
    
    # Load blacklist
    blacklist = load_blacklist()
    logger.info(f"Loaded {len(blacklist)} blacklisted machines")
    
    # Cleanup existing instances
    try:
        # Import here to avoid circular dependency
        import sys
        import importlib
        kill_module = importlib.import_module("utils.utils.kill_vast_ai_instances")
        logger.info("Cleaning up existing instances...")
        kill_module.kill_all_vastai_instances()
        time.sleep(5)  # Cooldown period
    except Exception as e:
        logger.warning(f"Failed to cleanup existing instances: {e}")
    
    # Search for available pods
    start_time = time.time()
    instance_id = None
    machine_id = None
    
    while time.time() - start_time < MAX_RETRY_TIME:
        # Get offers
        offers = get_offers()
        
        if not offers:
            logger.warning(f"No offers available. Retrying in {FIND_POD_SLEEP} seconds...")
            time.sleep(FIND_POD_SLEEP)
            continue
        
        # Filter offers by budget and blacklist
        filtered_offers = [
            pod for pod in offers
            if calculate_full_pod_cost(pod) <= BUDGET
            and pod.get("machine_id") not in blacklist
        ]
        
        if not filtered_offers:
            logger.warning(
                f"No offers within budget (${BUDGET}/hr) or all blacklisted. "
                f"Retrying in {FIND_POD_SLEEP} seconds..."
            )
            time.sleep(FIND_POD_SLEEP)
            continue
        
        # Sort by cost (cheapest first)
        filtered_offers.sort(key=lambda p: calculate_full_pod_cost(p))
        
        # Try to create instance with cheapest available pod
        for pod in filtered_offers:
            pod_id = pod.get("id")
            machine_id = pod.get("machine_id")
            cost = calculate_full_pod_cost(pod)
            
            # Reset instance_id for this iteration
            instance_id = None
            
            logger.info(f"Attempting to create instance on pod {pod_id} (machine {machine_id})")
            logger.info(f"Estimated cost: ${cost:.4f}/hour")
            
            # Get instances before creation for verification
            before_instances = get_all_instances()
            before_ids = {str(inst.get("id", "")) for inst in before_instances}
            logger.debug(f"Instances before creation: {list(before_ids)}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
            try:
                # Build startup command
                onstart_cmd = build_startup_command()
                
                # Vast.ai CLI expects --onstart to be a file path, not a command string
                # Write the command to a temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                    f.write(onstart_cmd)
                    onstart_file = f.name
                
                try:
                    # Create instance
                    # Use custom image if set, otherwise use default
                    image_to_use = os.getenv("VASTAI_DOCKER_IMAGE", DOCKER_IMAGE)
                    logger.info(f"Using Docker image: {image_to_use}")
                    
                    # Vast.ai CLI accepts --onstart for startup commands (as a file path)
                    cmd = [
                        "vastai", "create", "instance", str(pod_id),
                        "--image", image_to_use,
                        "--onstart", onstart_file,
                        "--disk", "30",
                        "--ssh"
                    ]
                    
                    logger.debug(f"Running command: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Log full output for debugging
                    logger.info(f"Command stdout: {result.stdout}")
                    if result.stderr:
                        logger.debug(f"Command stderr: {result.stderr}")
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(onstart_file)
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file {onstart_file}: {e}")
                
                # Parse instance ID from output
                output = result.stdout.strip()
                logger.info(f"Instance creation output: {output}")
                
                # Vast.ai CLI typically outputs instance ID or JSON
                parsed_instance_id = None
                try:
                    instance_data = json.loads(output)
                    parsed_instance_id = instance_data.get("id") or instance_data.get("new_contract") or instance_data.get("new_instance")
                    logger.debug(f"Parsed instance ID from JSON: {parsed_instance_id}")
                except json.JSONDecodeError:
                    # Try to extract ID from text output
                    # Could be just a number, or JSON-like text
                    if output:
                        # Try to find numeric ID in output
                        # Look for numeric IDs (could be at start, middle, or end)
                        numbers = re.findall(r'\d+', output)
                        if numbers:
                            # Take the last (likely largest) number as instance ID
                            parsed_instance_id = numbers[-1]
                            logger.debug(f"Extracted instance ID from numbers: {parsed_instance_id}")
                        else:
                            # Fallback to last word/token
                            parsed_instance_id = output.split()[-1] if output else None
                            logger.debug(f"Extracted instance ID from last token: {parsed_instance_id}")
                
                # Clean up instance ID - remove any non-numeric characters
                if parsed_instance_id:
                    # Extract only digits
                    cleaned_id = re.sub(r'\D', '', str(parsed_instance_id))
                    if cleaned_id:
                        parsed_instance_id = cleaned_id
                    else:
                        parsed_instance_id = None
                
                # Wait a moment for instance to appear in API
                time.sleep(2)
                
                # Get instances after creation
                after_instances = get_all_instances()
                after_ids = {str(inst.get("id", "")) for inst in after_instances}
                logger.debug(f"Instances after creation: {list(after_ids)}")
                
                # Verify the parsed instance ID exists
                if parsed_instance_id and parsed_instance_id in after_ids:
                    instance_id = parsed_instance_id
                    logger.info(f"Instance created and verified: {instance_id}")
                else:
                    # Try to find newly created instance by comparing before/after
                    new_id = find_newly_created_instance(before_instances, after_instances)
                    if new_id:
                        instance_id = new_id
                        logger.info(f"Found newly created instance (parsed ID was incorrect): {instance_id}")
                    else:
                        logger.error(f"Failed to verify instance creation. Parsed ID: {parsed_instance_id}")
                        logger.error(f"Output was: {output}")
                        logger.error(f"Before instances: {list(before_ids)}")
                        logger.error(f"After instances: {list(after_ids)}")
                        continue
                
                # Wait for instance to become running
                if wait_for_pod(instance_id, timeout=MAX_POD_WAIT):
                    # Final verification - make sure instance is still visible
                    verified = verify_instance_exists(instance_id)
                    if verified:
                        logger.info(f"Instance {instance_id} is ready and verified!")
                        
                        # Upload GCP credentials is now handled by copy_data_to_instance


                        # Log instance ID to status DB
                        try:
                            # Import here to avoid circular dependency
                            from utils.database.status_db import status_db
                            
                            # Get context info
                            run_id = os.getenv("AIRFLOW_RUN_ID", "manual")
                            dag_id = os.getenv("AIRFLOW_DAG_ID", "manual_dag")
                            task_id = os.getenv("AIRFLOW_TASK_ID", "vast_ai_train")
                            
                            status_db.log_event(
                                dag_name=dag_id,
                                task_name=task_id,
                                model_name="trl", # Assuming TRL model for now
                                run_id=run_id,
                                event_type="INSTANCE_CREATED",
                                status="RUNNING",
                                message=f"Created Vast.ai instance {instance_id}",
                                metadata={"instance_id": instance_id}
                            )
                        except Exception as e:
                            logger.error(f"Failed to log instance ID to status DB: {e}")

                        # Direct Data Upload Step
                        copy_data_to_instance(instance_id)
                        
                        return instance_id
                    else:
                        logger.warning(f"Instance {instance_id} became ready but is not visible in instance list.")
                        # Still return it, but try to copy data anyway
                        copy_data_to_instance(instance_id)
                        return instance_id
                else:
                    # Instance failed to start, kill it before trying next pod
                    logger.error(f"Instance {instance_id} failed to start. Killing instance and blacklisting machine {machine_id}")
                    try:
                        # Import here to avoid circular dependency
                        import sys
                        import importlib
                        kill_module = importlib.import_module("utils.utils.kill_vast_ai_instances")
                        kill_module.kill_instance(instance_id)
                        logger.info(f"Killed failed instance {instance_id}")
                    except Exception as e:
                        logger.warning(f"Failed to kill instance {instance_id}: {e}")
                    
                    # Blacklist machine
                    blacklist.add(machine_id)
                    save_blacklist(blacklist)
                    instance_id = None
                    continue
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to create instance on pod {pod_id}: {e}")
                if 'cmd' in locals():
                    logger.warning(f"Command: {' '.join(cmd)}")
                logger.warning(f"Error output (stderr): {e.stderr}")
                logger.warning(f"Standard output (stdout): {e.stdout}")
                # If subprocess failed, no instance was created, so nothing to clean up
                continue
            except Exception as e:
                logger.error(f"Unexpected error creating instance: {e}")
                # If we have an instance_id but hit an exception, try to kill it
                if instance_id:
                    try:
                        import sys
                        import importlib
                        kill_module = importlib.import_module("utils.utils.kill_vast_ai_instances")
                        kill_module.kill_instance(instance_id)
                        logger.info(f"Killed instance {instance_id} after unexpected error")
                    except Exception as kill_error:
                        logger.warning(f"Failed to kill instance {instance_id} after error: {kill_error}")
                continue
        
        # If we get here, all pods in this batch failed
        logger.warning(f"No pods available in this batch. Retrying in {FIND_POD_SLEEP} seconds...")
        time.sleep(FIND_POD_SLEEP)
    
    logger.error("Failed to create instance within retry time limit")
    return None


if __name__ == "__main__":
    # For testing
    instance_id = create_instance(DEBUG=True)
    if instance_id:
        print(f"Successfully created instance: {instance_id}")
    else:
        print("Failed to create instance")
