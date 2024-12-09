# QAGD_Pipeline.py

# Pipeline Architecture

# Constructor: Initializes the QAGD pipeline
# - Parameters:
#   - model: QAGD model, default is none
#   - tokenizer: Tokenizer, default is T5Tokenizer
#   - scheduler: QAGD scheduler, default is none
#   - device: Device to run the pipeline, default is 'cuda' if available else 'cpu'

# Call: Generates an output sequence (Answer) from an input sequence (Question) using the trained model
# - Parameters:
#   - prompt: Prompt (Question)
#   - num_steps: Number of steps to gradually add noise to the target data
#   - temperature: Temperature for sampling the output sequence
#   - max_seq_len: Maximum sequence length for the output sequence
#   - min_seq_len: Minimum sequence length for the output sequence
# - Pseduocode:
#   - Encode the prompt using the tokenizer
#   - Initialize the answer embedding with pure gaussian white noise
#   - For each step in num_steps (reverse diffusion steps):
#     - Predict the noise to remove using the model conditioned on the encoded prompt
#     - Update the answer embedding by removing the predicted noise
#   - Decode the denoised embedding into the output sequence using the tokenizer
#   - Return the generated answer sequence

# Function: from_pretrained
# - Parameters:
#   - model_path: Path to the trained model
#   - tokenizer_path: (Optional) Path to the tokenizer
# - Returns:
#   - QAGD pipeline with the pretrained model and tokenizer

# Function: encode_prompt
# - Parameters:
#   - prompt: Prompt (Question)

