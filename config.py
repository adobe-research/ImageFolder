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

project = 'ilo-train-p4de'
job_name = 'eval-var-0903'
num_nodes = 1
num_gpus = 8
job_file = 's3://xangl9867'
WANDB_API_KEY = 'eed03e9548474fc9bccb341783e5704c46647181'


with open('/Users/apple/.ssh/id_rsa_new', 'r') as file:
    private_key_content = file.read()