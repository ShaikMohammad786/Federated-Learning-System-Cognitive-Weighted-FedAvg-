# Federated Cognitive Contribution Learning (FCCL) for GPT-2  
*A Full Federated LLM Fine-Tuning System with FedIT + FedVA (DPO), LoRA Adapters, Non-IID Data, Weighted Aggregation & Real-Time Dashboard*

---

## 🧠 Overview

This project implements a **complete federated learning system** for fine-tuning GPT-2 using:

- **FedIT** – Federated Instruction Tuning (Supervised)  
- **FedVA** – Federated Value Alignment using **DPO**  
- **LoRA adapters** – Efficient parameter updates  
- **Non-IID Dirichlet client datasets**  
- **FCCL (Federated Cognitive Contribution Learning)** – Contribution-weighted aggregation  
- **Docker-based client/server orchestration**  
- **Streamlit dashboard** for real-time metrics visualization  
- **Evaluation pipeline** comparing base vs federated models  

The goal is to simulate a realistic decentralized environment where clients have **different data distributions**, **different losses**, and **varying skill quality**, and use FCCL to improve global model updates beyond standard FedAvg.

This is essentially a **mini FedLLM framework**, fully runnable and extremely modular.

---

## 📂 Project Structure

```
.
├── client.py          # Client-side worker: loads data, trains, logs metrics & contributions
├── orchestrator.py    # Server: coordinates rounds, aggregation, FCCL weighting
├── training.py        # Core local training loops (SFT + DPO)
├── utils.py           # LoRA helpers, model loading, weighted aggregation, logprob utilities
├── data_prep.py       # Non-IID split generator for SFT + VA datasets
├── evaluate.py        # Compute perplexity, preference accuracy, sample generations
├── dashboard.py       # Streamlit dashboard for rounds, losses, client weights
├── config.py          # Global configuration (model, FL, FCCL)
├── docker-compose.yml # Multi-container simulation (server + N clients + dashboard)
├── Dockerfile         # Base image for orchestrator and clients
└── data/
    └── clients/       # Auto-generated SFT + VA JSONL data shards per client
```

---

## 🏗️ System Architecture

```mermaid
flowchart TB

    subgraph DP[1. Data Preparation]
        DS1[Load Alpaca SFT Data]
        DS2[Load UltraFeedback / Synthetic VA Data]
        DS3[Dirichlet Non-IID Split (alpha = NON_IID_ALPHA)]
        DS1 --> DS3
        DS2 --> DS3
        DS3 --> C0_data[client_0_sft.jsonl + va.jsonl]
        DS3 --> C1_data[client_1_sft.jsonl + va.jsonl]
        DS3 --> C2_data[client_2_sft.jsonl + va.jsonl]
    end

    subgraph ORC[2. Orchestrator (Server)]
        O1[Start Round]
        O2[Write phase_R.txt (sft or va)]
        O3[Write global_round_R.pt (LoRA adapter)]
        O4[Wait for client done flags]
        O5[Collect client adapters + contribution JSON]
        O6[Compute FCCL Weights<br/>loss_inverse / data_size / combined]
        O7[Weighted Aggregation<br/>average_adapter_states_weighted()]
        O8[Save updated global adapter]
    end

    subgraph CL[3. Clients (Docker Workers)]
        Cn1[Load phase + global adapter]
        Cn2[Load client_i_sft or client_i_va data]
        Cn3[local_sft_train or local_dpo_train]
        Cn4[Compute avg_loss + sample_count]
        Cn5[Write adapter update<br/>client_i_round_R.pt]
        Cn6[Write contrib_client_i_round_R.json]
        Cn7[Write done_i_R.flag]
    end

    subgraph DB[4. Dashboard (Streamlit)]
        DB1[Read logs/progress.jsonl]
        DB2[Read logs/client_metrics.jsonl]
        DB3[Read contrib_client_i_R.json]
        DB4[Visualize rounds, losses, weights, heartbeats]
    end

    DP --> CL
    ORC <--> CL
    ORC --> DB
```

---

## 🧩 Key Components

### 1️⃣ Non-IID Data Preparation
- Dirichlet-based heterogeneous split  
- Produces per-client SFT + VA datasets  

### 2️⃣ Local Training (Client)
- SFT: supervised instruction tuning  
- DPO: preference-based value alignment  
- Logs: loss, sample count, contribution metadata  

### 3️⃣ FCCL Aggregation (Server)
- Computes contribution weights:  
  - `loss_inverse`  
  - `data_size`  
  - `data_loss_combined`  
- Softmax temperature for score sharpness  
- Weighted LoRA adapter aggregation  

### 4️⃣ Dashboard
- Round lifecycle  
- FCCL weights chart  
- Client losses over time  
- Live client heartbeat  
- Exchange artifact explorer  

### 5️⃣ Evaluation
- Perplexity  
- Preference accuracy  
- Sample generations  

---

## ⚙️ Running

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Prepare Data
```
python data_prep.py
```

### 3. Run with Docker
```
docker-compose up --build
```

### 4. Launch Dashboard
```
streamlit run dashboard.py
```

### 5. Evaluate
```
python evaluate.py --sft 50 --va 50 --gen-samples 3 --gen-tokens 30
```

---

## 🧠 FCCL Explained

FCCL replaces FedAvg with contribution-aware aggregation:

### Contribution score examples:
```
score = 1 / loss
score = num_samples
score = num_samples / loss
```

### Normalized weights:
```
softmax(score_i / T)
```

Lower temperature → sharper, more selective weighting.

---

## 📄 Resume Summary

- Built a complete federated GPT‑2 fine‑tuning system with LoRA, non‑IID clients, SFT + DPO training, Docker orchestration, and a real‑time FL dashboard.  
- Designed and implemented **Federated Cognitive Contribution Learning (FCCL)**: weighted aggregation using loss and data‑aware contributions.  
- Produced a full evaluation suite (perplexity, preference accuracy, qualitative generations) comparing baseline vs FCCL‑tuned GPT‑2.

---

## 📜 License
MIT License.
