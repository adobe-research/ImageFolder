"""
ADOBE CONFIDENTIAL
Copyright 2024 Adobe
All Rights Reserved.
NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
"""

import sys
import subprocess
from config import *
import os

def run_command(command, node_id, home_dir=True, timeout=None):
    home_dir = '/home/user'
    if node_id == 0:
        node_name = 'master-0'
    else:
        node_name = 'worker-' + str(node_id-1)

    init_commands = []
    init_commands.append('source /etc/profile')
    #init_commands.append('export CUDA_HOME=/opt/conda/envs/MFM_mim')
    #init_commands.append('export LD_LIBRARY_PATH=/opt/conda/envs/MFM_mim/lib64:$LD_LIBRARY_PATH')
    #init_commands.append('export OFI_NCCL_DISABLE_GDR_REQUIRED_CHECK=1')
    init_commands.append('source /opt/venv/bin/activate')
    init_commands = '; '.join(init_commands)

    command_prefix = f"""runai exec {job_name} --pod {job_name}-{node_name} -p {project} -- bash -c "{init_commands}; cd {home_dir} ; """
    command = command_prefix + command + '"'
    print(f"Executing command on {node_name}: {command}")

    if timeout is None:
        try:
            return subprocess.check_output(command, shell=True).decode(sys.stdout.encoding)
        except:
            return None
    else:
        try:
            return subprocess.check_output(command, shell=True, timeout=timeout).decode(sys.stdout.encoding)
        except subprocess.TimeoutExpired:
            return None

def setup_aws_credentials(i, aws_file='/Users/apple/Desktop/CMU/Debug/aws_credential_data.txt'):
    aws_cred_path = '~/.aws/credentials'
    run_command('mkdir -p ~/.aws', i)
    aws_creds = read_aws_credentials(aws_file)
    run_command('rm ~/.aws/credentials', i)
    run_command(f'echo [default] >> {aws_cred_path}', i)
    for c in aws_creds:
        run_command(f'echo {c} >> {aws_cred_path}', i)
    return aws_creds

def read_aws_credentials(aws_file='aws_credential_data.txt'):
    with open(aws_file) as f:
        aws_creds = f.readlines()
    aws_creds = [c.replace('\n', '') for c in aws_creds]
    return aws_creds
