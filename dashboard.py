"""Streamlit dashboard for monitoring FedIT/FedVA rounds and FCCL weights."""

# Run with: `streamlit run dashboard.py` (respects LOG_DIR and EXCHANGE_DIR env vars).

import glob
import json
import os
import time

import pandas as pd
import streamlit as st

from config import FED_ROUNDS

st.set_page_config(page_title="Federated LLM Dashboard", layout="wide")

# Directories shared with Docker containers; defaults allow local runs.
LOG_DIR = os.environ.get("LOG_DIR", "logs")
EXCHANGE_DIR = os.environ.get("EXCHANGE_DIR", "exchange")

st.title("Federated LLM Simulation")
st.write(
    "This dashboard visualizes FedIT (SFT) and FedVA (DPO) rounds plus FCCL "
    "client contribution weights derived from loss/data metadata."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Rounds")
    progress_path = os.path.join(LOG_DIR, "progress.jsonl")
    rounds_df = pd.DataFrame()
    if os.path.exists(progress_path):
        rows = [json.loads(l) for l in open(progress_path, "r", encoding="utf-8").read().splitlines() if l.strip()]
        if rows:
            rounds_df = pd.DataFrame(rows)
            if "event" not in rounds_df.columns:
                rounds_df["event"] = ""
            rounds_df = rounds_df.sort_values(["round", "event"]).reset_index(drop=True)
            st.dataframe(rounds_df, use_container_width=True, height=240)

            # Overall progress across SFT and VA (total = 2 * FED_ROUNDS rounds)
            try:
                total_planned_rounds = int(2 * FED_ROUNDS)
            except Exception:
                total_planned_rounds = 2
            aggregated_so_far = int((rounds_df.get("event") == "aggregated").sum())
            overall_pct = min(1.0, aggregated_so_far / max(total_planned_rounds, 1))
            st.subheader("Overall Progress (SFT + VA)")
            st.progress(overall_pct, text=f"{aggregated_so_far} / {total_planned_rounds} rounds aggregated")

            # Build per-round lifecycle view from exchange artifacts
            lifecycle = []
            for _, row in rounds_df.iterrows():
                r = int(row.get("round", -1))
                phase = row.get("phase", "?")
                selected = row.get("selected_clients", []) or []
                sel_set = set(int(x) for x in selected)
                # Broadcast exists if global of this round is present
                global_path = os.path.join(EXCHANGE_DIR, f"global_round_{r}.pt")
                broadcasted = os.path.exists(global_path)
                # Client updates for selected only
                client_updates = []
                done_flags = []
                for cid in sel_set:
                    client_updates.append(os.path.exists(os.path.join(EXCHANGE_DIR, f"client_{cid}_round_{r}.pt")))
                    done_flags.append(os.path.exists(os.path.join(EXCHANGE_DIR, f"done_{cid}_{r}.flag")))
                received = int(sum(1 for x in client_updates if x))
                done_cnt = int(sum(1 for x in done_flags if x))
                total = max(len(sel_set), 1)
                completion = round(100.0 * done_cnt / total, 1)
                # Aggregation is inferred when next global file appears (or final done phase exists)
                next_global = os.path.exists(os.path.join(EXCHANGE_DIR, f"global_round_{r+1}.pt"))
                aggregated = bool(next_global)
                lifecycle.append({
                    "round": r,
                    "phase": phase,
                    "selected": sorted(list(sel_set)),
                    "broadcast": broadcasted,
                    "client_updates": received,
                    "done_flags": done_cnt,
                    "% complete": completion,
                    "aggregated": aggregated,
                })

            contrib_rows = []
            for _, row in rounds_df.iterrows():
                if row.get("event") == "aggregated":
                    weights = row.get("contribution_weights") or {}
                    if isinstance(weights, dict):
                        for cid, weight in weights.items():
                            contrib_rows.append({
                                "round": int(row.get("round", -1)),
                                "phase": row.get("phase", "?"),
                                "client": int(cid),
                                "weight": float(weight),
                            })

            if lifecycle:
                st.subheader("Round Lifecycle")
                lifecycle_df = pd.DataFrame(lifecycle).sort_values("round")
                st.dataframe(lifecycle_df, use_container_width=True, height=220)

                # Render compact progress bars per round
                st.subheader("Per-Round Progress")
                for _, row in lifecycle_df.iterrows():
                    r = int(row["round"])
                    label = f"Round {r} ({row['phase']})"
                    pct = float(row["% complete"]) / 100.0
                    st.progress(pct, text=label)

            st.subheader("FCCL – Client Contribution Weights")
            if contrib_rows:
                contrib_df = pd.DataFrame(contrib_rows).sort_values(["round", "client"]).reset_index(drop=True)
                st.dataframe(contrib_df, use_container_width=True, height=220)
                latest_round = contrib_df["round"].max()
                latest = contrib_df[contrib_df["round"] == latest_round]
                if not latest.empty:
                    st.bar_chart(latest.set_index("client")["weight"])
            else:
                st.info("Waiting for aggregated contribution weights...")
    else:
        st.info("Waiting for rounds...")

with col2:
    st.subheader("Client Losses")
    client_path = os.path.join(LOG_DIR, "client_metrics.jsonl")
    client_df = pd.DataFrame()
    if os.path.exists(client_path):
        rows = [json.loads(l) for l in open(client_path, "r", encoding="utf-8").read().splitlines() if l.strip()]
        if rows:
            client_df = pd.DataFrame(rows)
            st.dataframe(client_df, use_container_width=True, height=300)
            try:
                chart_df = client_df.groupby(["phase", "client"]).tail(1)
                st.line_chart(client_df, x=None, y="loss", color="phase")
            except Exception:
                pass
    else:
        st.info("Waiting for client metrics...")

st.subheader("Exchange Artifacts")
if os.path.exists(EXCHANGE_DIR):
    phases = sorted([p for p in os.listdir(EXCHANGE_DIR) if p.startswith("phase_")])
    globals_ = sorted([p for p in os.listdir(EXCHANGE_DIR) if p.startswith("global_round_")])
    clients = sorted([p for p in os.listdir(EXCHANGE_DIR) if p.startswith("client_")])
    st.write({"phases": len(phases), "globals": len(globals_), "client_updates": len(clients)})
    # Live Client Status
    status_files = sorted(glob.glob(os.path.join(EXCHANGE_DIR, "status_*.json")))
    if status_files:
        statuses = []
        now = time.time()
        for p in status_files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                    rec["age_s"] = round(now - rec.get("ts", now), 1)
                    statuses.append(rec)
            except Exception:
                pass
        if statuses:
            st.subheader("Live Client Status")
            st.dataframe(pd.DataFrame(statuses).sort_values(["round", "client"]).reset_index(drop=True), use_container_width=True, height=220)

    # VA reference indicator
    ref_path = os.path.join(LOG_DIR, "ref_model_sft.pth")
    if os.path.exists(ref_path):
        st.info("VA reference ready: logs/ref_model_sft.pth present (produced after SFT).")
else:
    st.info("Waiting for exchange directory...")

st.caption("Refresh every few seconds to update. Streamlit can also be set to auto-refresh.")
