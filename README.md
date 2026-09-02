# langchain-rag

Tutor virtual de disciplina baseado em RAG (LangChain + LangGraph). Responde
dúvidas de alunos usando exclusivamente o material fornecido pelo professor,
com abordagem pedagógica — evita respostas diretas, guia o raciocínio em
vez de entregar a solução de cara — e cita a fonte exata (arquivo, capítulo,
seção) de onde tirou cada informação.

## Como funciona

O projeto tem duas aplicações separadas:

- **App do professor** (Streamlit — `app.py`): único propósito é criar a
  base de conhecimento de uma disciplina a partir de PDFs (slides ou
  texto), extraindo estrutura (capítulos/seções) e indexando num vector
  store (Chroma). Não faz chat — isso é só ferramenta de preparação,
  usada antes do deploy.
- **App do aluno** (Chainlit — `chainlit_app.py`): tela de login por
  matrícula (validada contra uma lista de matrículas do professor), chat
  com sidebar de conversas anteriores por aluno, e logout. Cada deploy
  serve **uma única disciplina** — é a base gerada pela etapa acima que
  fica embarcada na imagem.

Por trás dos dois, o mesmo agente: um grafo LangGraph
(`agent/graph.py`) com 5 nós —
`tracking → planning → (retrieve, condicional) → assess_documents → generate_answer`.
Ele mantém um perfil de aprendizagem do aluno entre turnos, decide uma
estratégia pedagógica antes de gerar qualquer resposta, e num único passo
por trecho recuperado (`assess_documents`) pontua a relevância e extrai a
evidência estruturada — só responde citando essa evidência
(anti-alucinação). Cada nó usa um modelo diferente, configurável por
variável de ambiente — modelos baratos/rápidos para tracking/grading,
melhores para planning/geração.

## Rodando localmente

```bash
pip install -r requirements.txt
cp .env.example .env  # preencha OPENAI_API_KEY e CHAINLIT_AUTH_SECRET (gerado com `chainlit create-secret`)
```

**Professor** — criar uma base de conhecimento a partir de PDFs:

```bash
streamlit run app.py
# ou via CLI, sem UI:
python -m scripts.create_kb <pasta_de_pdfs> "<nome da disciplina>"
```

**Aluno** — conversar com o tutor (requer exatamente uma disciplina em
`data/knowledge_bases/` e uma `data/roster.txt` com as matrículas
autorizadas):

```bash
chainlit run chainlit_app.py -w --port 8501
```

### Variáveis de ambiente

| Variável | Obrigatória | Padrão | Descrição |
|---|---|---|---|
| `OPENAI_API_KEY` | sim | — | LLMs e embeddings |
| `CHAINLIT_AUTH_SECRET` | sim (app do aluno) | — | assinatura do login; gerar com `chainlit create-secret` |
| `ROSTER_PATH` | não | `data/roster.txt` | lista de matrículas autorizadas |
| `EMBED_MODEL` | não | `text-embedding-3-large` | modelo de embedding |
| `GENERATION_MODEL` / `PLANNING_MODEL` / `TRACKING_MODEL` / `GRADING_MODEL` | não | ver `agent/chat_pipeline.py` | um modelo por nó do grafo |
| `MODEL_TEMPERATURE` | não | `0` | temperatura dos modelos acima |
| `COURSE_LEVEL` | não | `beginner` | nível instrucional |
| `ANSWER_LANGUAGE` | não | `Português` | idioma das respostas |
| `ALLOW_DIRECT_ANSWERS` | não | `True` | permite o aluno pedir resposta direta, pulando o passo-a-passo pedagógico |

## Deploy

Cada disciplina vira uma imagem Docker própria (base de conhecimento e
roster embarcadas). Veja [DEPLOY.md](DEPLOY.md) para o passo a passo
completo, incluindo o deploy usado em produção (Railway).

## Estrutura

```
agent/     grafo LangGraph, prompts, estado do agente
rag/       ingestão de PDF → parsing → chunking → vector store
utils/     roster, helpers de parsing, citações
app.py                app Streamlit do professor (criar disciplina)
chainlit_app.py        app Chainlit do aluno (chat)
main.py                CLI mínima pra testar o agente sem UI
scripts/create_kb.py   CLI pra criar uma base de conhecimento sem UI
```

## Tipos de Documentos Reconhecidos

- Slides
- PDFs textuais
