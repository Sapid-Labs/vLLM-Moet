# Session handoffs

One file per investigation, numbered by attempt: `NN-brief-title.md`.
A fresh session starts by reading the highest-numbered file for its topic.

Each file tracks, in this order:

1. **Target** — the metric or outcome that ends the investigation.
2. **State** — what works right now (serving config, artifacts on disk).
3. **Attempts log** — numbered, append-only: what was tried, what happened,
   what it ruled out. Never delete entries; a later session must be able to
   see every dead end.
4. **Established facts** — measurements and root causes with evidence.
5. **Next steps** — ranked, with concrete commands.
6. **Resume mechanics** — scripts, gotchas, and any venv-vs-repo drift.

When a session ends (or an investigation pivots), update the file — or
create `NN+1-*.md` if the goal itself changed. Keep the narrative history in
`~/Dev/howtospark/models/*.md` (lab log); keep the *actionable resume state*
here.
