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

Go through "Criar disciplina", upload the course's PDFs, and give it a
name.

**CLI**: point the script at a folder of PDFs directly — no UI needed,
good for scripting/automating a teacher's pipeline end-to-end.

```
python -m scripts.create_kb <pdf_folder> "<discipline name>"
```

Both produce the same thing: a folder under `data/knowledge_bases/<kb_id>/`.

**Before building the image, make sure `data/knowledge_bases/` contains
exactly one course** — `chainlit_app.py` expects a single knowledge base
and will refuse to start otherwise. If you've been testing multiple
courses locally, move the others out of the way first.

## 2. Prepare the roster

Create `data/roster.txt`: one valid enrollment ID per line. Blank lines
and lines starting with `#` are ignored.

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
