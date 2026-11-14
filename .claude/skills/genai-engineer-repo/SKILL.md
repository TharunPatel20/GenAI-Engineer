---
name: genai-engineer-repo
description: Use when adding or editing a project, the AI-Engineer-Guide, a session recap, START-HERE, a README, or a Colab notebook in the GenAI Engineer study repo, or when checking the repo for consistency. Triggers on requests to add a project, update guide sections, fix a notebook, change learning-path links, or review cross-links and source tags.
---

# GenAI Engineer Study Repo

## 1. What this repo is

A GenAI engineer course study repository (Cohort 4). It contains:
- `AI-Engineer-Guide.md`: the concept guide, one section per schedule topic.
- `claude-files/`: per-session recaps named `recap-YYYY-MM-DD.md`.
- `START-HERE.md`: the single entry point and learning path.
- `projects/`: nine hands-on Colab-first projects (01 to 09) and an index in `projects/README.md`.
- `recording transcript/`: git-ignored Zoom transcripts.

Source of truth for topics is the schedule spreadsheet `100 Next Gen AI Cohort 4 Schedule.xlsx`. Recaps supply what was discussed. Do not add topics that are in neither.

## 2. Repo structure

| Path | Role |
|---|---|
| `README.md` | Repository index, topic list, where to start |
| `START-HERE.md` | Learning path (13 steps), Colab setup, key setup, progress checklist |
| `AI-Engineer-Guide.md` | Sections 1 to 13 plus "Notes on Sources" |
| `claude-files/recap-*.md` | One recap per session, linked from the guide and START-HERE |
| `projects/README.md` | Projects table, suggested path, status note |
| `projects/NN-name/README.md` | Per-project guide (see section 4) |
| `projects/NN-name/notebook.ipynb` | Colab notebook; Step 0 clones the repo if needed |
| `projects/NN-name/requirements.txt` | Exact version pins |
| `projects/NN-name/.env.example` | Placeholder key only |

## 3. Hard rules

- **Colab-first.** Each project has `notebook.ipynb` with an "Open in Colab" badge pointing to `https://colab.research.google.com/github/TharunPatel20/GenAI-Engineer/blob/main/projects/NN-name/notebook.ipynb`. Step 0 clones `TharunPatel20/GenAI-Engineer` only if the folder does not exist, then `chdir`s into the project. Keep no dead branches (no unused alternative code paths in the notebook).
- **Keys.** Keys come from Colab Secrets (`GEMINI_API_KEY`, read with `google.colab.userdata.get`) or from a local `.env` that is never committed. Never write a real key anywhere. `.env.example` holds only `your-key-here`.
- **Write and static-check only.** When editing, do not run code, install packages, or execute notebooks. Validate notebook JSON with a JSON parser. Check imports against `requirements.txt` by reading.
- **Source-tag the guide.** Every claim in `AI-Engineer-Guide.md` carries `[Recap]`, `[Schedule]`, or `[General]`, as the "Notes on Sources" section defines. Mark garbled recap content and do not propagate it.
- **Do not invent topics.** Corrections come from the schedule and the existing recaps only.
- **Label example output.** Example outputs are marked illustrative unless captured from a real run. Never imply reproducibility.
- **Transcripts.** Nothing in `recording transcript/` gets links or recordings added.
- **Git.** Do not commit unless asked. Do not push unless asked.

## 4. Standard project README structure

Use these section names, in this order, copied from `projects/01-python-foundations/README.md`:
1. Title (`# Project NN: ...`), Colab badge, one-line description
2. Difficulty and estimated time
3. Learning Objectives
4. Prerequisites
5. Run in Google Colab
6. Setup
7. Build Steps (numbered; each step has **What**, **Why**, and **Code**)
8. Expected Output and Checkpoints
9. Diagram (Mermaid or ASCII only; no external images)
10. Troubleshooting (table: Problem, Likely cause, Fix)
11. Stretch Goals
12. What You Learned
13. References (link to the recap and to the guide section)

Each project folder also holds `notebook.ipynb`, `requirements.txt` (exact pins), and `.env.example` (placeholder only). Offline-capable projects say so in the README and label demo output as an offline demo.

## 5. Cross-link expectations

- Use relative links only.
- The guide, recaps, `projects/README.md`, and `START-HERE.md` must link to each other correctly. START-HERE links guide sections and recaps; each project links back to its guide section and recap.
- Guide anchors follow GitHub slug rules: lowercase, punctuation removed, spaces become hyphens, and the section number comes first. Example: `## 1. Python Foundations (Prerequisite Refresher)` becomes `#1-python-foundations-prerequisite-refresher`.
- Verify every relative link resolves to an existing file, including the anchor where one is used.

## 6. Workflow with sub-agents

- Use one sub-agent per project for audits or edits. Give each the project folder path, this skill's rules, and the narrow task.
- Each sub-agent returns one line: what changed or what it found.
- The main session reviews every change before anything is committed.
- Sub-agents do not commit or push.

## 7. Checklist before finishing a change

- [ ] All relative links resolve to existing files (and guide anchors match slugs)
- [ ] No placeholders left unreplaced (for example `TODO`, `NN-name`, or `Replace me` where real content is required)
- [ ] No real keys, tokens, or Zoom links anywhere; `.env.example` holds only `your-key-here`
- [ ] Each `.ipynb` parses as valid JSON; Step 0 and the Colab badge match the project folder
- [ ] Imports in the notebook appear in `requirements.txt`, checked by reading
- [ ] Guide claims carry source tags; garbled recap content is marked
- [ ] Example outputs are labelled illustrative unless captured from a real run
- [ ] `git status` shows only the intended files; nothing under `recording transcript/` changed
- [ ] Nothing committed or pushed unless the user asked
