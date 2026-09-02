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
  first boot. It holds the student app's `chainlit.db` (chat
  history/thread sidebar) — without it, every restart wipes all
  conversations.
- The app comes up at `http://localhost:8501` (or whatever host/port
  you're forwarding to).

## Deploying to Railway (instead of plain Docker)

Railway builds this same Dockerfile. Deploy from the repo root with:

```
railway up --no-gitignore
```

**`--no-gitignore` is required.** `railway up` respects `.gitignore`,
which excludes `data/` wholesale (the vector store is rebuilt locally,
not versioned) — so without the flag `data/knowledge_bases/` and
`data/roster.txt` never reach the build context and `COPY` fails with
`"/data/roster.txt": not found`. With the flag, `.railwayignore` is the
single source of truth for what to exclude (it already drops `venv/`,
`.env`, `data/chats/`, `.files/`, `pdfs/`, etc.).

The volume must be mounted at `/app/data/chats` (same reasoning as the
`-v` note above). `OPENAI_API_KEY` and `CHAINLIT_AUTH_SECRET` are set as
service variables in the Railway dashboard, not passed on the command
line.

## Multiple courses

Repeat steps 1-4 per course: a fresh knowledge base, a fresh
`data/roster.txt`, a distinct image tag, and a distinct host port
(`-p 8502:8501`, `-p 8503:8501`, ...). Each is a fully independent
container — there's no shared state or shared roster between them.

## Updating a course's material mid-semester

There's no live "add a PDF" flow in the container. To update the
material: re-run step 1 to rebuild the knowledge base (or add PDFs and
recreate it), rebuild the image, and redeploy. Existing chat history in
the volume is untouched by this since it's stored separately.

## Class metrics (on demand)

The student app records one anonymous row per conversation turn — topic,
comprehension/progress estimate, frustration, pedagogical strategy,
which parts of the material were used — into `metrics.db`, next to
`chainlit.db` on the mounted volume (`/app/data/chats/`). No enrollment
id, no thread id: rows are independent events, so you can see "the class
is stuck on topic X" but not follow an individual student.

The file is created on first boot and lives on the volume, so it survives
redeploys and restarts. It is **not** in the container image — only the
code that writes it is. Pull the file whenever you want to look, then
open it in the local app.

**Railway.** `railway volume files` copies over SSH, so once per machine
you need a key registered with Railway:

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
