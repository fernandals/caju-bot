# langchain-rag

Tutor virtual de disciplina baseado em RAG (LangChain + LangGraph). Responde
dúvidas de alunos usando exclusivamente o material fornecido pelo professor,
com abordagem pedagógica — guia o raciocínio com perguntas em vez de
entregar a solução de cara — e cita as páginas do material de onde tirou
cada informação, com um link que abre o PDF na página.

## Como funciona

O projeto tem duas aplicações separadas:

- **App do professor** (Streamlit — `app.py`): cria a base de
  conhecimento de uma disciplina a partir de PDFs (slides ou texto) e
  mostra as métricas de uso. Não faz chat — é ferramenta de preparação,
  usada antes do deploy.
- **App do aluno** (Chainlit — `chainlit_app.py`): login com matrícula
  (validada contra a lista de matrículas da turma) e nome, chat com a
  barra lateral de conversas anteriores, e logout pelo menu do avatar.
  Cada deploy serve **uma única disciplina** — a base gerada pela etapa
  acima fica embarcada na imagem.

Por trás dos dois, o mesmo agente: um grafo LangGraph (`agent/graph.py`).
Um pedido de lista de assuntos é respondido direto, a partir da lista de
tópicos da disciplina (`agent/outline.py`). Os demais turnos passam por
`tracking → planning → retrieve → assess_documents → generate_answer`, e
cumprimentos e conversa fiada pegam um atalho (`greet`) depois do
tracking. O agente mantém um `LearningState` do aluno entre turnos e
decide a estratégia pedagógica antes de gerar qualquer resposta. A busca
traz os 8 trechos mais parecidos com a pergunta. Depois, num único passo
por trecho (`assess_documents`), um modelo pontua a relevância e extrai a
evidência estruturada, e o tutor só responde citando essa evidência
(anti-alucinação). Cada nó usa um modelo configurável por variável de
ambiente.

O ensino segue um arco por tópico (`agent/teaching.py`, por regras em
código): perguntas guiadas, explicação completa quando o aluno trava,
uma recapitulação com uma pergunta de checagem e, quando o aluno mostra
que entendeu, o **tópico concluído** — o tutor confirma, recapitula e
oferece o próximo assunto ou um exercício, sem reabrir o tópico. Quem
decide se o aluno entendeu é o material: quando o aluno responde a uma
pergunta, uma chamada curta (`judge_reply` em `agent/nodes.py`) avalia a
resposta contra a evidência — correta, parcial, errada ou sem tentativa —
antes de o tutor escrever. Uma resposta correta fecha o tópico (a partir
da 2ª resposta do arco, ou logo na 1ª depois de uma explicação ou num
exercício); nas outras, o tutor recebe o veredito e responde de acordo.
Quem diz que tem prova recebe a explicação direto, uma vez por tópico;
depois disso as respostas são avaliadas como as de qualquer aluno.

Além das explicações guiadas, o tutor:
- monta um **roteiro de revisão da disciplina inteira**, com citações para
  cada assunto, quando o aluno pede para revisar para a prova sem dizer o
  tema;
- **corrige a resposta do aluno** a um exercício que ele mesmo passou —
  e um exercício resolvido certo conclui o tópico;
- formata fórmulas em LaTeX e código em Markdown.

O tutor ensina em um de três **estilos**, que moldam todas as respostas,
das perguntas guiadas à explicação (`agent/prompts.STYLE_INSTRUCTIONS`):
**passo a passo** (o entendimento construído em degraus pequenos — o
padrão), **com exemplos** (casos concretos do material ou analogias do
dia a dia) e **com exercícios** (pequenas tarefas com valores concretos
para o aluno resolver). O aluno escolhe na primeira conversa e pode
trocar quando quiser em **Configurações** (ícone no cabeçalho). O estilo
também se ajusta sozinho, por regra em código (`agent/style_tracking.py`):
cada arco que termina — tópico dominado ou aluno travado — conta para o
estilo em uso, e depois de **3 arcos travados seguidos** o tutor passa
para o estilo com melhor histórico. O tutor nunca troca de estilo sem
essa evidência ou o pedido do aluno. Fica salvo por matrícula
(`style_trackers` em `data/chats/student_profiles.db`) e vale também no
meio de uma conversa.

Além do estado por conversa, o app do aluno mantém um **perfil de
aprendizagem por matrícula** entre sessões (`agent/profiler.py`,
`utils/student_profile.py`): se responde
bem a perguntas socráticas, tendência a frustração, tópicos já dominados
/ ainda difíceis, concepções erradas que já expressou, o que funcionou com
ele, o que já sabe de fora da disciplina, e uma nota curta pro tutor. É
atualizado **uma vez por sessão** — no início de cada conversa, um modelo
relê a conversa anterior do aluno (e os sinais por turno do tutor) e
mescla no perfil (persistido em `data/chats/student_profiles.db`). A
personalização observada liga a partir da 2ª conversa.
O perfil entra no prompt de geração e, por regras em código (sem chamada
de modelo), ajusta o ritmo e o plano da resposta: limite de perguntas
guiadas por aluno/tópico (sempre ao menos 2 antes de explicar), tópico já
dominado fecha mais cedo, tópico difícil ganha explicação passo a passo
(`agent/teaching.py`, `agent/plan_profile.py`,
`agent/familiarity.py`); o que o perfil sabe
sobre o tópico atual vai junto da instrução do turno
(`agent/profile_notes.py`). O perfil nunca é alterado durante a
conversa (o estilo, sim — ver acima). Ao
retomar uma conversa pela barra lateral, o tutor continua de onde parou,
com as citações e os botões da última resposta.

## Ingestão do material

Funciona para qualquer disciplina — as regras usam só a estrutura dos
PDFs, nada específico de um curso:

- **Slides ou documento**: decidido por três sinais (página deitada,
  fonte grande para a largura da página, poucas palavras por página);
  basta dois para ser slide, então um deck exportado em retrato também
  é reconhecido.
- **Slides**: passos de animação (o slide seguinte repete o anterior com
  mais itens) são descartados, e slides seguidos são agrupados em trechos
  de até 200 tokens. A citação mostra a faixa de páginas.
- **Documentos**: divididos pelos títulos numerados (`13.1 Título`).
- **Títulos**: o título de cada slide e de cada arquivo vem da maior
  fonte da página; o cabeçalho da instituição, repetido em vários
  arquivos, é ignorado.

PDFs escaneados (sem texto) são avisados na criação da base, mas não
passam por OCR; fórmulas desenhadas como imagem se perdem na extração.

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

**Avaliar a busca** de uma disciplina — por exemplo, antes e depois de
mudar a ingestão. Gera uma vez um conjunto de perguntas a partir dos PDFs
e compara bases: a atual e uma gerada com `--out-dir` numa pasta que o
app não carrega:

```bash
python -m evals.retrieval_eval generate <pasta_de_pdfs> data/evals/<disciplina>.json
python -m scripts.create_kb <pasta_de_pdfs> "<nome da disciplina>" --out-dir /tmp/kb_teste
python -m evals.retrieval_eval run data/evals/<disciplina>.json data/knowledge_bases/<kb_id> /tmp/kb_teste/<kb_id>
```

**Testes** (sem chamada de modelo, sem chave de API):

```bash
pip install -r requirements-dev.txt
pytest
```

### Variáveis de ambiente

| Variável | Obrigatória | Padrão | Descrição |
|---|---|---|---|
| `OPENAI_API_KEY` | sim | — | LLMs e embeddings |
| `CHAINLIT_AUTH_SECRET` | sim (app do aluno) | — | assinatura do login; gerar com `chainlit create-secret` |
| `ROSTER_PATH` | não | `data/roster.txt` | lista de matrículas autorizadas |
| `EMBED_MODEL` | não | `text-embedding-3-large` | modelo de embedding |
| `GENERATION_MODEL` / `PLANNING_MODEL` / `TRACKING_MODEL` / `GRADING_MODEL` | não | ver `agent/chat_pipeline.py` | um modelo por nó do grafo |
| `REVIEW_MODEL` | não | `gpt-4.1-mini` | avalia a resposta do aluno a uma pergunta contra o material (decide se o tópico está concluído) e escreve a resposta que confere a tentativa num exercício |
| `PROFILER_MODEL` | não | `gpt-4.1-mini` | modelo do profiler de aluno (roda 1× por sessão) |
| `PERSONALIZATION_MODE` | não | `alternate` | estudo de personalização: `alternate` (conversas alternam com/sem perfil por aluno), `always` ou `off` — ver DEPLOY.md |
| `TITLE_MODEL` | não | `gpt-4.1-mini` | gera o título curto da conversa na barra lateral (1× por conversa) |
| `MODEL_TEMPERATURE` | não | `0` | temperatura dos modelos acima |
| `MODEL_TIMEOUT` | não | `30` | timeout (s) por chamada de LLM |
| `MODEL_MAX_RETRIES` | não | `2` | tentativas por chamada de LLM |
| `COURSE_LEVEL` | não | `beginner` | nível instrucional |
| `ANSWER_LANGUAGE` | não | `Português` | idioma das respostas |
| `ALLOW_DIRECT_ANSWERS` | não | `True` | permite o aluno pedir resposta direta, pulando o passo-a-passo pedagógico |

### Aparência

Tudo em `public/`, carregado pelo `.chainlit/config.toml`:

- `login.html` — a página de login do app (substitui a do Chainlit);
- `app.js` — leva o Chainlit até essa página e só abre o PDF de uma
  citação quando o aluno clica nela;
- `style.css` — citações legíveis e os três pontinhos de "pensando";
- `avatars/tutor_virtual.png` (ícone das respostas — o nome vem de
  `name` na configuração), `favicon.png` (aba do navegador) e
  `tutor.png` (página de login). Para trocar, substitua mantendo os nomes.

O seletor de estilo fica no painel de Configurações do Chainlit, no
cabeçalho (`chat_settings_location = "sidebar"`).

O navegador guarda esses arquivos em cache: ao mudar `style.css` ou
`app.js`, troque o `?v=` deles no `.chainlit/config.toml`.

## Deploy

Cada disciplina vira uma imagem Docker própria (base de conhecimento e
roster embarcadas). Veja [DEPLOY.md](DEPLOY.md) para o passo a passo
completo, incluindo o deploy usado em produção (Railway).

## Estrutura

```
agent/     grafo LangGraph, prompts, estado, perfil de aluno, lista de tópicos
rag/       ingestão de PDF → parsing → chunking → vector store, e a busca
utils/     roster, citações, formatação, métricas, perfil de aluno
public/    página de login, script, estilos e ícones do app do aluno
evals/     avaliação da busca e do tracking (chamam os modelos)
tests/     testes sem chamada de modelo
app.py                  app Streamlit do professor (criar disciplina + métricas)
chainlit_app.py         app Chainlit do aluno (chat)
main.py                 CLI mínima pra testar o agente sem UI
scripts/create_kb.py    CLI pra criar uma base de conhecimento sem UI
scripts/pull_metrics.py CLI pra baixar metrics.db do volume do Railway
scripts/export_study.py CLI pra exportar o estudo de personalização em CSV
```
