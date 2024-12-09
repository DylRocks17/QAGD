# QAGD_diffuser.py

# Model Architecture

# Encoder Architecture
# - Uses transformer encoder to encode the input sequence (Question) 

# Decoder Architecture
# - Uses transformer decoder to decode the output sequence (Answer concatenated with Context)

# Forward Process
# - Gradually adding noise to the target data (Answer concatenated with Context) over several steps transforming them into noise distributions

# Reverse Process
# - Learning to iteratively remove the noise, conditioned on input (Question), to reconstruct or generate coherent and contextually relevant answers

import torch
import torch.nn as nn
from math import pi as PI

from QAGD_Schedulers import QAGD_Scheduler
from logger import get_logger

# Question-Answer Generative Diffuser Class
class QAGD_Diffuser(nn.module):
    logger = get_logger(__name__)

    # Constructor
    def __init__(self, scheduler: QAGD_Scheduler):
        super(QAGD_Diffuser, self).__init__()
        self.scheduler = scheduler

    # Forward Diffusion Step
    def forward(self, answer: str) -> str:
        return # noisy_input

    # Reverse Diffusion Step 
    def reverse(self, question: str) -> str:
        return # denoised_output
    

