"""Global configuration for the FCCL federated LLM demo."""

MODEL_NAME = "gpt2"   # small causal model
NUM_CLIENTS = 3
CLIENTS_PER_ROUND = 3                # sample available clients each round (like paper)
FED_ROUNDS = 1                       # make small (paper used 100-200)
SFT_LOCAL_EPOCHS = 1
VA_LOCAL_EPOCHS = 1
SFT_LR = 5e-5
VA_LR = 5e-5
LORA_R = 8                            # small rank for LoRA in mini demo
LORA_ALPHA = 16
BATCH_SIZE = 2
MAX_LENGTH = 128
SEED = 42
DATA_SFT_SPLIT = "train[:200]"        # tiny slice
DATA_VA_SPLIT = "train[:200]"         # tiny slice
NON_IID_ALPHA = 0.3                   # Dirichlet concentration for non-IID client splits

# FCCL / contribution-aware aggregation
CONTRIBUTION_MODE = "cognitive"          # "uniform" (FedAvg) vs "cognitive" (FCCL weighting)
CONTRIBUTION_SOURCE = "loss_inverse"     # 'loss_inverse', 'data_size', or 'data_loss_combined' scoring basis
CONTRIBUTION_TEMPERATURE = 1.0           # Softmax temperature over scores; lower => peakier weights