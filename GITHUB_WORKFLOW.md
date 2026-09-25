# Team GitHub Workflow — Amazon ML Challenge 2026

This file explains exactly how the 4 of you share code and track submissions through GitHub, on top of what `WORKPLAN.md` already assigns to each track. Read this once at the start; you shouldn't need to think about git again after that if you follow it.

---

## 0. One-time setup (Person D / whoever has the code right now)

1. Go to github.com → **New repository** → make it **Private** → do NOT initialize with a README (you already have a local repo).
2. In your `student_resource/` folder on your PC:
   ```
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```
3. GitHub → your repo → **Settings → Collaborators → Add people** → add the other 3 by their GitHub username or email.
4. Send them the repo URL (or they'll get a GitHub invite email — they must accept it).

## 1. One-time setup (everyone else — Persons A, B, C)

1. Accept the GitHub invite email (or the link Person D sends you).
2. Clone the repo:
   ```
   git clone https://github.com/<username>/<repo-name>.git
   cd <repo-name>
   ```
3. Get the actual dataset separately (it's NOT in git — too big). Download it from wherever the team got it originally and place it at `dataset/train/` and `dataset/test/` inside your cloned folder, matching the structure already described in `README.md`.
4. Create your own track branch (pick the one matching your assigned track from `WORKPLAN.md` Section 3):
   ```
   git checkout -b track-a-blocking      # Person A
   git checkout -b track-b-model         # Person B
   git checkout -b track-c-validation    # Person C
   ```
   Person D stays on `main`.

## 2. Daily workflow (Persons A, B, C)

You only ever touch your own files (see the ownership rules in `WORKPLAN.md` Section 4 — "Merge conflict prevention rules"). Work, commit, and push to **your own branch** as often as you like:

```
git add <your files, e.g. src/blocking.py>
git commit -m "short description of what changed"
git push origin <your-branch-name>
```

You do **not** need to open a Pull Request or wait for review — Track D merges your branch directly at the two scheduled integration points (`WORKPLAN.md` Section 4). Just make sure your branch is pushed and up to date before those times (Day 2 ~13:00 IST and ~20:00 IST).

If you want Track D to look at something before the scheduled integration, open a Pull Request (`git push` then GitHub will show a "Compare & pull request" button) so there's a clear diff to review — otherwise a plain push is enough.

## 3. Integration workflow (Person D only)

At each integration point:
```
git checkout main
git pull origin main                  # make sure your local main is current
git merge track-a-blocking
git merge track-b-model
# resolve any conflicts (WORKPLAN.md Section 4 tells you where conflicts are likely)
git push origin main
```
Then run the full pipeline (train → predict → validate) per `WORKPLAN.md` Section 4, and decide go/no-go for the next leaderboard submission.

## 4. Tracking leaderboard submissions in git

Every time you upload `matching_results.tsv` to the portal, commit a record of it so the whole team can see the history without anyone having to ask "what's our score now":

1. Open `WORKPLAN.md`, find the **Submission log** table in Track D's section, and fill in the row for that submission number (date/time, code state, val F0.5, leaderboard F0.5, notes).
2. Commit just that change:
   ```
   git add WORKPLAN.md
   git commit -m "Submission #N: LB F0.5=0.XXXX"
   git push origin main
   ```
This gives you a searchable git history of every submission — `git log --oneline -- WORKPLAN.md` shows the whole submission timeline at a glance.

Do **not** commit the actual `dataset/`, `output/`, `sample_dense/`, or `models/` folders — they're already excluded by `.gitignore`, keep it that way (they're too big and are regenerated locally by anyone who clones the repo).

## 5. Daily status updates

Same rule as above — Track D (or whoever updates last each day) commits the 5-line status update block in `WORKPLAN.md` Section 5 for all 4 people at the end of each day:
```
git add WORKPLAN.md
git commit -m "Day N status updates"
git push origin main
```
Everyone pulls (`git pull origin main`) before starting work the next day to see it.

## 6. Quick command reference

| I want to... | Command |
|---|---|
| Get the latest `main` (e.g. after an integration) | `git checkout main && git pull origin main` |
| Switch back to my own track branch | `git checkout track-a-blocking` (or your track name) |
| Pull my track branch's latest changes onto `main` after Track D merged it | `git checkout main && git pull origin main` |
| See what changed recently | `git log --oneline -10` |
| See who changed what in a file | `git blame <file>` |
| Undo a local uncommitted change | `git checkout -- <file>` |

## 7. If something goes wrong

- **Merge conflict during integration:** only Track D resolves these, and only in the 2–3 files where overlap is expected (see `WORKPLAN.md` Section 4). If a conflict shows up somewhere unexpected, it means someone edited a file outside their assigned ownership — stop, message the team, and figure out who touched what before force-resolving.
- **Accidentally committed a huge dataset file:** don't push it. If you haven't pushed yet: `git reset --soft HEAD~1`, fix `.gitignore`, recommit. If you already pushed, tell Person D immediately — it needs a proper history rewrite, don't try to fix it solo.
- **Not sure if you're on the right branch:** `git branch` (the one with `*` next to it is your current branch).
