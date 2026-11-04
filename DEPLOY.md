# Deploying a course as a container

One container per course. Each image bakes in exactly one course's
knowledge base and roster, so a teacher with multiple courses builds and
runs one image per course (different tags, different ports).

**Important:** the roster is an allowlist gate, not real authentication —
the login screen's "name" field is not checked against anything, and
anyone who knows or guesses a valid enrollment ID can see that student's
chat history. That's an acceptable tradeoff for a low-stakes pilot, not
something to rely on for anything sensitive.

## 1. Prepare the course knowledge base

Two ways to do this — pick whichever fits your workflow:

**UI**: ingest PDFs through the existing dev app.

```
streamlit run app.py
```

Go through "Criar disciplina", upload the course's PDFs and the SIGAA
grade sheet (`.xls`), and give it a name. This produces both the
knowledge base and `data/roster.txt` in one step — skip step 2 below.

**CLI**: point the script at a folder of PDFs directly — no UI needed,
good for scripting/automating a teacher's pipeline end-to-end.

```
python -m scripts.create_kb <pdf_folder> "<discipline name>" --roster notas.xls
```

`--roster` is optional; when given, it also writes `data/roster.txt` from
the SIGAA grade sheet's "Matrícula" column (step 2).

Both produce the same thing: a folder under `data/knowledge_bases/<kb_id>/`.

**Before building the image, make sure `data/knowledge_bases/` contains
exactly one course** — `chainlit_app.py` expects a single knowledge base
and will refuse to start otherwise. If you've been testing multiple
courses locally, move the others out of the way first.

## 2. Prepare the roster

If you used the "Criar disciplina" UI in step 1, this is already done —
it read the SIGAA grade sheet and wrote `data/roster.txt` for you.

Otherwise, create `data/roster.txt` by hand: one valid enrollment ID per
line. Blank lines and lines starting with `#` are ignored.

```
# Software Architecture - Fall 2026
20261001
20261002
20261003
```

## 3. Build the image

```
docker build -t tutor-<course-slug> .
```

## 4. Run it

```
docker run -d \
  -p 8501:8501 \
  -e OPENAI_API_KEY=sk-... \
  -e CHAINLIT_AUTH_SECRET=... \
  -v tutor-<course-slug>-chats:/app/data/chats \
  --name tutor-<course-slug> \
  tutor-<course-slug>
```

- `OPENAI_API_KEY` is required and is **not** baked into the image — pass
  it at run time. Don't commit it anywhere.
- `CHAINLIT_AUTH_SECRET` is also required (login won't work without it) —
  generate one with `chainlit create-secret` and pass it at run time, not
  baked into the image either.
- The `-v` volume must be mounted at `/app/data/chats` specifically, not
  `/app/data` — the latter would shadow the baked-in
  `data/knowledge_bases/` and `data/roster.txt` behind an empty volume on
  first boot. It holds three SQLite files, all created on first boot:
  `chainlit.db` (chat history / thread sidebar), `metrics.db` (per-turn
  metrics + personalization-study sessions, keyed by enrollment ID), and `student_profiles.db` (per-student learning
  profile, keyed by enrollment ID). Without the volume, every restart
  wipes all of them.
- The app comes up at `http://localhost:8501` (or whatever host/port
  you're forwarding to).

## Deploying to Railway (instead of plain Docker)

Railway builds this same Dockerfile straight from the GitHub repo: every
push to `main` redeploys. The service settings live in the Railway
dashboard, not in the repo: Railway's Config as Code (`railway.toml`) is
deprecated and new services can no longer opt in, so there is no config
file here on purpose.

The build only sees what's **versioned** — that's why `.gitignore` lets
`data/knowledge_bases/` and `data/roster.txt` through while ignoring the
rest of `data/`. Push the course's knowledge base and roster to `main`
before the first deploy, or `COPY` fails with
`"/data/roster.txt": not found`.

One-time setup in the Railway dashboard:

1. **New service → GitHub Repo**, pick this repository and the `main`
   branch (the first time, Railway asks to install its GitHub app and to
   be given access to the repo).
2. **Variables**: `OPENAI_API_KEY` and `CHAINLIT_AUTH_SECRET` (required).
   Don't set `PORT` — Railway injects it and the container listens on it.
3. **Volume**: right-click the service → *Attach volume*, mount path
   `/app/data/chats`.
4. **Networking → Generate Domain** (HTTP) for the public URL. Railway
   detects the port on its own; if it asks, it's `8501`.
5. **Settings**:
   - *Build → Builder*: **Dockerfile**. Railpack, the default, would
     guess the start command and pick the wrong app (`app.py`/`main.py`).
     Leave *Custom Build Command*, *Start Command* and the pre-deploy step
     empty: the Dockerfile's `CMD` already starts Chainlit on `PORT`.
   - *Regions & Replicas*: **US East**, **1 replica**. The chats,
     profiles and metrics are SQLite files on the volume, and a volume
     attaches to a single replica; Chainlit sessions also live in that
     one process's memory.
   - *Healthcheck Path*: `/`, so a build that can't start shows up as a
     failed deploy instead of a live but broken one.
   - *Restart Policy*: **On Failure**, 5 retries.
   - Leave *Serverless*, *CDN Caching*, *Cron Schedule* and *Wait for CI*
     off. Serverless makes the first visitor wait for a cold start, CDN
     caching can serve stale or someone else's pages, and there's no
     GitHub Actions workflow for *Wait for CI* to wait on.

The volume must be mounted at `/app/data/chats` (same reasoning as the
`-v` note above). The model/timeout knobs (`GENERATION_MODEL`, `PROFILER_MODEL`,
`MODEL_TIMEOUT`, …) are optional service variables — see
`.env.example`/README for defaults.

## Multiple courses

Repeat steps 1-4 per course: a fresh knowledge base, a fresh
`data/roster.txt`, a distinct image tag, and a distinct host port
(`-p 8502:8501`, `-p 8503:8501`, ...). Each is a fully independent
container — there's no shared state or shared roster between them.

## Reusing a deployment (and its volume) for a different course

If you rebuild the image with a different course and deploy it onto the
same volume, students' stored profiles carry over only what describes the
learner (explanation style and its arc record, how they respond to
guiding questions, frustration tendency). The topic lists and the tutor note
belong to a course: they are cleared the first time the profile is loaded
under the new course. Conversations are tagged with their course, and the
profiler only reads conversations from the current one. The study panel
and `scripts.export_study --discipline` also compare one course at a time.

## Updating a course's material mid-semester

There's no live "add a PDF" flow in the container. To update the
material: re-run step 1 to rebuild the knowledge base (or add PDFs and
recreate it), rebuild the image, and redeploy. Existing chat history in
the volume is untouched by this since it's stored separately.

The knowledge base is versioned (a GitHub deploy builds from it), but
Chroma rewrites its files whenever the app opens them - so running the
app locally showed `chroma.sqlite3` as modified, ready to be committed by
mistake. Git is told to ignore those local changes; the flag lives in
your checkout, not in the repo, so set it once per clone:

```
git ls-files -z data/knowledge_bases | grep -z /chroma/ | xargs -0 git update-index --skip-worktree
```

When you rebuild the knowledge base on purpose, lift it before
committing (and set it again after):

```
git ls-files -z data/knowledge_bases | grep -z /chroma/ | xargs -0 git update-index --no-skip-worktree
```

## Class metrics (on demand)

The student app records one row per conversation turn — topic,
comprehension/progress estimate, frustration, pedagogical strategy,
which parts of the material were used — into `metrics.db`, next to
`chainlit.db` on the mounted volume (`/app/data/chats/`).

Since 2026-10-06 the rows are **identifiable**: each carries the
student's enrollment ID, the conversation (thread) id, the turn number
and the id of the tutor's reply (to join the 👍/👎 stored in
`chainlit.db`). This is for the personalization study below. No names —
the login name is free text. Rows written before that date keep NULLs in
these columns; the schema is migrated in place on boot.

### Personalization study

`PERSONALIZATION_MODE` (service variable) decides whether the tutor uses
each student's profile:

| Value | Behavior |
|---|---|
| `always` (default) | every conversation personalized — for testing the personalization itself |
| `alternate` | the study design: each student's conversations alternate **with** / **without** personalization; whether a student starts "with" or "without" is a stable hash of the enrollment ID, so half the class starts each way. Set it when the study starts |
| `off` | never personalized |

"Without" only hides the profile from the tutor — the profiler keeps
updating it from every conversation. A resumed conversation keeps its
arm. Each conversation's arm and a snapshot of the student's profile at
its start are stored in `metrics.db` (`study_sessions` table).

To compare, pull **both** files (`python -m scripts.pull_metrics
--chainlit`) and open the metrics tab — the "Estudo de personalização"
section compares the two conditions, overall and per student against
themselves. For your own analysis:

```
python -m scripts.export_study
```

writes `data/study/sessions.csv` (one row per conversation, with its
outcomes), `arms.csv` and `paired.csv`. These contain enrollment IDs —
keep them out of anything you publish without anonymizing.

The file is created on first boot and lives on the volume, so it survives
redeploys and restarts. It is **not** in the container image — only the
code that writes it is. Pull the file whenever you want to look, then
open it in the local app.

The same volume also holds `student_profiles.db` — the per-student
learning profile the tutor uses to personalize itself. Unlike
`metrics.db`, this one **is** identifiable (keyed by enrollment ID); it
stores only slow-moving learning-style traits and a short free-text note,
never transcripts. It's refreshed once per session by a model that reads
the student's previous conversation. The same file holds
`style_trackers`: each student's explanation style (step by step,
examples, exercises) and how their guided arcs went in each, updated by
code after each turn (`agent/style_tracking.py`). There's no viewer for it — pull it
with `railway volume files ... download /student_profiles.db ...` if you
need to inspect or reset it.

**Easiest — the helper script.** From the repo root, with the Railway
project linked (`railway status` should show the `langchain-rag`
service):

```
python -m scripts.pull_metrics
```

This downloads `metrics.db` from the volume to `data/chats/metrics.db`
(overwriting any previous copy) — exactly where the metrics tab looks by
default. Then:

```
streamlit run app.py
```

and open the **📊 Métricas da turma** tab.

Useful flags:

- `--open` — run `streamlit run app.py` automatically once the download
  finishes.
- `--chainlit` — also pull `chainlit.db` (chat history) into the same
  folder.
- `--volume NAME` / `--dest PATH` — override the volume name or the local
  destination.

The script shells out to the Railway CLI, so the one-time SSH key setup
below still applies.

**Manual — the Railway CLI directly.** `railway volume files` copies over
SSH, so once per machine you need a key registered with Railway:

```
railway ssh keys add            # walks you through generating/registering a key
```

Then, from the linked project directory (the volume is mounted *at*
`/app/data/chats`, so inside the volume the file sits at the root):

```
railway volume files --volume langchain-rag-volume list /
railway volume files --volume langchain-rag-volume download /metrics.db ./data/chats/metrics.db --overwrite
```

No SSH key set up? The Railway dashboard also has a volume file browser
(service → **Volume**) you can download `metrics.db` from directly.

**Plain Docker:**

```
docker cp tutor-<course-slug>:/app/data/chats/metrics.db ./data/chats/metrics.db
```

Then:

```
streamlit run app.py
```

and open the **📊 Métricas da turma** tab (it defaults to
`data/chats/metrics.db`, or upload the file you just downloaded).
