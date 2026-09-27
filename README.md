#Amazon ML Challenge 2026: Business Entity Resolution

Team: Aditya Ajeeth (Team Leader), Pratham Rampurmath, Adithya Sundar, Aayushman Singh.

| Where | What |
|---|---|
| `code/business_entity_resolution/` | the runnable pipeline: `src/`, `tests/`, `README.md` (exact reproduction commands), pinned `requirements.txt`, `ARCHITECTURE.md` |
| `ML_chads_Documentation.md` | the methodology write-up (the filled-in challenge template; named `Documentation_template.md` inside the submission zip, as the rules require) |
| `problem_statement.md` | the challenge description |
| `utils/validate_submission.py` | the organisers' submission validator |
| `archive/` | development notes, planning docs and one-off analysis scripts kept for history; not needed to reproduce |

Final model: blocking + two-stage LightGBM + expected-F0.5 decoder, trained on a sample that keeps the real
competing businesses; public leaderboard macro F0.5 **0.953**. No external data, APIs or pretrained models.
