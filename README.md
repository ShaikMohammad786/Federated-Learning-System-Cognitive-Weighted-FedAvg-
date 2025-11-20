# Federated LLM Instruction Tuning with Cognitive Contribution Learning

Federated Cognitive Contribution Learning (FCCL) extends classic FedAvg by weighting each client's update according to its *cognitive contribution* (loss/data quality). This repo simulates federated LoRA fine-tuning of GPT-2 over instruction-tuning (FedIT/SFT) and value alignment (FedVA/DPO) phases using Dockerized orchestrator/clients, a shared exchange directory, and a Streamlit monitoring dashboard.

## Architecture
- **`data_prep.py`** – builds per-client non-IID datasets with a Dirichlet splitter.
- **`orchestrator.py`** – coordinates FedIT + FedVA rounds, broadcasts adapters, waits for clients, and aggregates with FCCL weights.
- **`client.py`** – runs local SFT/VA updates, logs metrics, and writes adapters + contribution metadata.
- **`dashboard.py`** – Streamlit UI showing round lifecycle, client losses, and FCCL weights from `logs/progress.jsonl`.
- **`training.py`** – shared local training loops for SFT and DPO.
- **`evaluate.py`** – compares base GPT-2 vs global FCCL-tuned model (perplexity, preference accuracy, generations).
- **`legacy/main.py`** – single-process FedAvg baseline (kept for reference only).

## Key Features
- Non-IID client splits via configurable Dirichlet concentration (`NON_IID_ALPHA`).
- Two-phase FedIT (SFT) + FedVA (DPO) loop with shared exchange directory communication.
- FCCL weighting sourced from per-client loss/data stats (`CONTRIBUTION_SOURCE`) with temperature scaling.
- Streamlit dashboard tracking rounds, live client statuses, and contribution bars per round.
- Docker Compose stack for orchestrator + multiple clients + dashboard; also runnable as local scripts.
- Evaluation script for perplexity, preference accuracy, and qualitative generations.

## How to Run
### 1. Data Prep
```bash
python data_prep.py
```

### 2. Option A – Docker Compose
```bash
docker compose up --build orchestrator client
docker compose up dashboard        # in another terminal to keep logs clean
# visit http://localhost:8501 for the dashboard
```

### 3. Option B – Local Processes
```bash
python orchestrator.py
CLIENT_ID=0 python client.py
CLIENT_ID=1 python client.py
# ...repeat per client ID (can share same machine)
streamlit run dashboard.py
```

## Configuration (`config.py`)
- `NUM_CLIENTS`, `CLIENTS_PER_ROUND`, `FED_ROUNDS`: federation settings.
- `NON_IID_ALPHA`: Dirichlet concentration; lower => more skewed per-client data.
- `CONTRIBUTION_MODE`: `"uniform"` (FedAvg) or `"cognitive"` (FCCL).
- `CONTRIBUTION_SOURCE`: `"loss_inverse"`, `"data_size"`, or `"data_loss_combined"`.
- `CONTRIBUTION_TEMPERATURE`: softmax temperature over contribution scores (lower = sharper).
Edit these constants before running to toggle FCCL behaviors.

## Evaluation
After training completes and `logs/global_final.pth` exists:
```bash
python evaluate.py --sft 100 --va 100 --gen-samples 5
```
Outputs include base vs fine-tuned perplexity, preference accuracy, and generated samples from the FCCL model.

## Dashboard Insights
Open the Streamlit app to monitor:
- Rounds table + overall completion bar.
- Per-round lifecycle (%) and exchange artifacts.
- Client loss history.
- **FCCL – Client Contribution Weights** table + latest-round bar chart (correlate with losses).

## FCCL Weighting Snippet
`orchestrator.py` turns per-client loss/data stats into aggregation weights:
```python
scores = []
for cid in selected:
    info = contrib_info.get(cid, {})
    loss_val = float(info.get("avg_loss") or 0.0)
    num_samples = float(info.get("num_samples") or 0.0)
    if CONTRIBUTION_SOURCE == "loss_inverse":
        base = 1.0 / max(loss_val, 1e-6)
    elif CONTRIBUTION_SOURCE == "data_size":
        base = num_samples
    elif CONTRIBUTION_SOURCE == "data_loss_combined":
        base = num_samples / max(loss_val, 1e-6)
    scores.append(base)
weights = torch.softmax(torch.tensor(scores) / CONTRIBUTION_TEMPERATURE, dim=0)
global_adapter = average_adapter_states_weighted(local_states, weights.tolist())
```
Switching `CONTRIBUTION_MODE` back to `"uniform"` reverts to classic FedAvg.
