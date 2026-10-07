# Campaign 7 Oct 2026, 01:42 -> 11:00 UTC+3

Goal (user): many applied, important settings where MG beats ALL analogs; literature on
monitoring simplification during training and using it to improve training; use subagents;
bring promising settings to confirmation; keep searching. Last update before submission.
Heartbeat cron 32b8748b fires at :13 and :43 until 10:43.

## Rules
AGENT_GUIDE.md (protocol before runs, calibration/test, all competitors, honesty).
CPU: <= 3 workers per agent, 4 agents max; RAM shared ~9 GB.

## Workstreams
| id | setting | owner | status |
|---|---|---|---|
| L | literature: monitoring simplification/collapse & acting on it | agent lit | DONE 02:10 -> literature/LIT_ru.md (ranked proposals: games cycles 30%, rule transfer 25%, offline-RL ckpt 15%, Cuttlefish/Early-Bird switch 12%, GAN 2D 10%) |
| S1 | loss of plasticity in continual learning: trigger resets/ReDo by MG | agent plasticity | started 01:50; hint sent: task-boundary level jumps favour MG |
| S3 | convergence/stationarity diagnostic: when to decay LR (Pflug, SASA, plateau) | agent lrdecay | started |
| S2 | SSL dimensional collapse: label-free hyperparameter/checkpoint choice (RankMe) | agent ssl | started |
| S4 | loss-spike precursors in small transformers | agent spikes | started 01:55 |
| G | cycles in learning in games (count live cycles; PSRO size / damping) | agent games | started 02:12 (2 workers) |
| T | monitor rule transfer across heterogeneous setups | agent transfer | started 02:12 (2 workers) |
| S9 | when to stop iterative pruning | queued | |
| R3 | offline-RL (neural FQI) checkpoint selection | queued | |

## Results log
(append here)

- 02:2x-03:0x: all 6 experiment agents stopped on the API session limit (reset 05:00).
  lrdecay `main.py all` kept running. 05:16: all six resumed via SendMessage with context,
  asked to be token-economical (quota may run out again before 11:00).
