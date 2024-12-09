# QAGD_Trainer.py

# Trainer Architecture

# Constructor: Initializes the QAGD trainer
# - Parameters:
#   - model: QAGD model, default is none (allows user to continue training from a checkpoint)
#   - tokenizer: Tokenizer, default is T5Tokenizer
#   - scheduler: QAGD scheduler, default is none
#   - device: Device to run the trainer, default is 'cuda' if available else 'cpu'

# Function: Train Model
# - Parameters:
#   - train_loader: DataLoader for training data
#   - val_loader: DataLoader for validation data 
#   - num_epochs: Number of epochs to train the model
#   - num_steps: Number of steps to gradually add noise to the target data
#   - learning_rate: Learning rate for the optimizer
#   - weight_decay: Weight decay for the optimizer
#   - warmup_steps: Warmup steps for the scheduler
# - Returns:
#   - Trained QAGD model checkpoint, training loss, validation loss
# - Pseduocode:
#   - Loop over the number of epochs:
#     - Initialize epoch loss
#     - Loop over each batch in training data:
#       - Extract input questions from batch
#       - Extract target answers from batch
#       - Tokenize and encode the input questions and target answers
#       - Initialize answer embeddings
#       - Loop over the num_steps:
#         - Add noise to the answer embeddings (forward step)
#         - Predict the noise to remove (reverse step)
#         - Compute loss between predicted denoised embeddings and actual noise.
#         - Backpropagate the loss.
#         - Update the model parameters using the optimizer.
#         - Accumulate training loss
#       - Compute the average training loss for the epoch
#       - Set model to evaluation mode.
#       - Initialize validation loss.
#       - Disable gradient computation.
#       - Loop over each batch in val_loader:
#           - Extract input questions and answers from the batch.
#           - Encode the input questions using the tokenizer.
#           - Tokenize and encode the answers.
#           - Generate answer embeddings.
#           - Apply the forward diffusion process to corrupt the answer embeddings.
#           - Forward pass through the model to predict denoised embeddings.
#           - Compute loss between predicted denoised embeddings and actual noise.
#           - Accumulate validation loss.
#       - Computer the average validation loss for the epoch. 
#       - Log the training and validation losses.
#       - Re


# Dataset Preparation

# Preprocessing

# Model Training

