# Project Log — Agente  Educacional

## 🎯 Objetivo Atual
Construir um agente educacional que atue como **monitor inteligente de uma disciplina**, capaz de:
- Utilizar exclusivamente os materiais fornecidos pelo professor;
- Responder apenas dentro do domínio de determinado pelos documentos;
- **Não fornecer respostas diretas**, atuando de forma pedagógica e orientativa.

# ➡️ Log Entries

## 📅 Data
- **2026-01-27**

### 📌 Status do Projeto
Primeira versão funcional de um sistema RAG educacional baseada em PDFs, com pipeline completo de ingestão, recuperação e geração controlada de respostas.  
A implementação atual é considerada **dummy**, porém já valida a arquitetura geral pretendida para o projeto.

### ⚠️ Limitações Conhecidas

- Não há:
  - Persistência de embeddings;
  - Metadados estruturais (capítulos, seções, tipo de documento);
  - Diferenciação entre PDFs textuais e PDFs de slides;
  - Mecanismo de avaliação da qualidade da resposta pedagógica.
- O campo documents no estado ainda não é explorado de forma significativa.
- Recuperação baseada apenas em similaridade textual simples.
- Respostas do modelo são repetitivas
- O agente funciona apenas para SysADL, se trocar os PDFs ele buga

### 🔜 Próximos Passos

- Evoluir o processo de ingestão:
  - Classificação de tipo de documento (artigo, slide, capítulo);
  - Enriquecimento de metadados por chunk.
- Refinar o controle pedagógico:
  - Melhor separação entre decisão de recuperação e decisão pedagógica.
- Persistência do vector store.
- Avaliação do comportamento do agente com alunos reais ou cenários simulados.

---

## 📅 Data
- **2026-08-17**

### 📌 Status do Projeto
Arquitetura evoluiu significativamente desde o MVP inicial descrito acima. O pipeline de ingestão agora é estruturado em estágios (`rag/loader.py` → `rag/parser.py` → `rag/splitter.py` → `rag/vectorstore.py`), com persistência real do vector store (Chroma) por disciplina em `data/knowledge_bases/<kb_id>/chroma`, indexado por um `metadata.json` (nome da disciplina, contagem de chunks, modelo de embedding, etc.).

O agente foi reescrito como um grafo LangGraph (`agent/graph.py`) com 6 nós:
`tracking → planning → (retrieve, condicional) → grade_documents → extract_evidence → generate_answer`.

- `tracking`: infere e atualiza incrementalmente um `LearningState` (tópico, intenção, nível de compreensão, frustração) sem resetar a cada turno — o estado anterior é tratado como baseline.
- `planning`: decide uma estratégia pedagógica estruturada (`AnswerPlan`: strategy/depth/exemplos/analogias/exercícios/necessidade de retrieval) antes de qualquer geração de resposta.
- `retrieve`: monta query adaptativa (pergunta + tópico/subtópico + boosts por intenção/estratégia) e busca no Chroma via MMR (k=5, fetch_k=20).
- `grade_documents`: um LLM pontua cada chunk recuperado (0–1) por relevância pedagógica e filtra (threshold 0.5, com fallback para top-3).
- `extract_evidence`: converte cada chunk sobrevivente em evidência estruturada (`ChunkEvidence`) com citação exata derivada dos metadados (ex.: `[Section: X, Pages: Y-Z]`), proibido de responder a pergunta diretamente.
- `generate_answer`: gera a resposta final seguindo o plano, citando apenas a evidência extraída, com regras explícitas anti-alucinação e proibição de expor prompts/ferramentas/estado interno.

Cada nó usa um modelo diferente (`agent/models.py: ModelRegistry`), configurável via variáveis de ambiente — permite usar modelos baratos/rápidos para tracking/grading e modelos melhores para planning/generation.

A aplicação Streamlit (`app.py`, `pages/create_kb.py`, `pages/chat.py`) já suporta múltiplas disciplinas: o professor cria uma KB nomeada a partir de PDFs enviados, e o aluno seleciona qual disciplina quer consultar. Conversas são persistidas por chat_id + disciplina em `data/chats/*.json`.

### ✅ Itens antes listados como limitação e já resolvidos
- Persistência de embeddings → Chroma persistido em disco por KB.
- Metadados estruturais (capítulos, seções, tipo de documento) → `rag/parser.py` extrai seções via regex e organiza blocos semânticos por parágrafo; `utils/helpers.detect_pdf_type` diferencia slides de PDF textual pela proporção da página.
- Diferenciação entre PDFs textuais e de slides → implementada (`parse_slides` vs `parse_pdf`).
- O campo de documentos recuperados agora é explorado de forma significativa: passa por grading e extração de evidência estruturada antes da geração, em vez de ir direto para o prompt final.

### ⚠️ Limitações ainda presentes (observadas no código atual)
- Ainda não há mecanismo de avaliação automática da qualidade pedagógica da resposta.
- `main.py` (entrada CLI) continua hardcoded para a disciplina "Software Architecture" e a pasta `pdfs/` — não foi atualizado para o padrão multi-KB usado no Streamlit.
- `grade_documents` e `extract_evidence` fazem uma chamada de LLM por chunk recuperado, sequencialmente (sem batching/paralelismo) — pode ficar lento quando há muitos documentos.
- Heurísticas de parsing continuam simples: detecção de título (`extract_document_title`) usa apenas a primeira linha não vazia da página 1; detecção de seções (`SECTION_REGEX`) depende de um padrão específico de numeração + espaçamento, potencialmente frágil para PDFs com layout diferente do material atual (SAIA).
- Não há suíte de testes automatizada integrada ao repositório (há um `test.py` na raiz, mas está fora do controle de versão/git status untracked).
- Código da primeira versão (`rag_minimal.py`, pasta `extra/`) ainda presente no repositório, aparentemente não utilizado pela aplicação atual (`app.py`/`pages/`).

### 🔜 Próximos Passos (revisão)
- Mecanismo de avaliação da qualidade pedagógica da resposta.
- Atualizar `main.py` para usar o mesmo fluxo multi-disciplina do Streamlit (ou removê-lo/consolidar com `agent/chat_pipeline.py`).
- Paralelizar/batchar as chamadas de grading e extração de evidência.
- Avaliação do comportamento do agente com alunos reais ou cenários simulados.

---

## 📅 Data
- **2026-08-24**

### 📌 Status do Projeto
App do aluno reescrito de Streamlit para **Chainlit** (`chainlit_app.py`, substitui `student_app.py`) — o motivo foi a interface Streamlit ser sentida como limitante pra um layout de chat moderno (login → chat com histórico lateral → logout). O backend (grafo LangGraph, pipeline RAG, roster) não mudou; só a camada de UI do fluxo do aluno. O app de professor (`app.py` + `pages/`) continua em Streamlit, intocado.

Login agora pede **matrícula + nome**: reaproveita a tela de login nativa do Chainlit (dois campos prontos), sem construir nada do zero — o campo "usuário" carrega a matrícula (validada contra `data/roster.txt`, mesma lógica de antes) e o campo "senha" foi realocado para capturar o nome do aluno (não é senha de verdade, não é checada contra nada). Textos relabelados via `.chainlit/translations/pt-BR.json` ("Matrícula"/"Nome completo"/"Entrar").

Sidebar de conversas anteriores agora é a **sidebar nativa de threads do Chainlit** (clique pra retomar, "nova conversa"), em vez do sistema de arquivos JSON usado por `utils/chat_ui.py`/`student_app.py`. Isso exigiu um `SQLAlchemyDataLayer` (SQLite, `data/chats/chainlit.db`) — o schema oficial do Chainlit é desenhado pra Postgres (`UUID`/`JSONB`/`TEXT[]`), então foi adaptado pra colunas `TEXT` e validado empiricamente (CRUD direto + teste via WebSocket) antes de virar código final; o app de professor continua usando o JSON antigo (`utils/helpers.save_chat`/`load_chats`) sem mudanças.

`_highlight_citations`/`_CITATION_PATTERN`, que viviam "privados" dentro de `utils/chat_ui.py`, foram extraídos para `utils/citations.py` (`highlight_citations` pública) — reaproveitada por `chat_ui.py` (Streamlit) e `chainlit_app.py` (Chainlit) sem duplicar a regex.

Deploy: app já estava rodando no Railway (`railway up`); a troca de framework subiu junto com a criação de um **volume persistente montado em `/app/data/chats`** — lacuna pré-existente (histórico se perdia a cada redeploy) que foi corrigida como parte dessa mudança. O mount é deliberadamente em `data/chats/`, não em `data/` inteiro, porque um volume vazio ali esconderia `data/knowledge_bases/` e `data/roster.txt`, que são gravados na imagem no build.

### ✅ Itens antes listados como limitação e já resolvidos
- `test.py`/pasta `extra/` (código da primeira versão, `rag_minimal.py`) mencionados como presentes-mas-não-usados no log anterior — já não existem no repositório.

### ⚠️ Limitações/trade-offs observados nesta rodada
- Chainlit traz `literalai` como dependência obrigatória, que por sua vez puxa ~80 pacotes de instrumentação OpenTelemetry (Anthropic, Cohere, Pinecone, etc. — nada disso é usado aqui). Não tem como evitar mantendo o Chainlit; infla `requirements.txt`/build, mas não deve pesar em runtime (bibliotecas paradas).
- Resumir uma conversa antiga pela sidebar continua "com perda", igual no Streamlit: só a lista de mensagens é restaurada, `student_profile`/`learning_state`/`teaching_state`/`evidence` resetam para o estado inicial (`on_chat_resume` em `chainlit_app.py`).
- `main.py` continua hardcoded pra "Software Architecture" (ver limitação de 08-17, ainda não endereçada).

### 🧹 Limpeza geral
- Removido `setup.py` (código quebrado — chamava `.mkdir()` numa `str` —, sem nenhuma referência no projeto).
- Removido `scripts/load_kb.py` (script de debug isolado, sem uso documentado).
- `scripts/create_kb.py` finalizado como CLI de verdade (`python -m scripts.create_kb <pasta> "<disciplina>"`), sem mais caminho absoluto hardcoded de uma máquina específica — passa a ser o pipeline pretendido pro professor gerar uma KB sem precisar da UI antes de buildar a imagem Docker.
- Removidos os 24 arquivos de tradução default do Chainlit não utilizados (só `pt-BR.json` é carregado, já que `language = "pt-BR"` é forçado em `.chainlit/config.toml`) — `.gitignore` ganhou uma regra pra evitar que voltem a ser versionados (o Chainlit os recria localmente a cada `chainlit run`, é comportamento do framework).
- `.mypy_cache/` (138MB) adicionado ao `.gitignore` — nunca esteve tracked, mas também nunca foi ignorado explicitamente.

### 🔜 Próximos Passos (revisão)
- Mecanismo de avaliação da qualidade pedagógica da resposta (ainda pendente desde 08-17).
- Atualizar `main.py` para o padrão multi-KB (ainda pendente desde 08-17).
- Paralelizar/batchar grading e extração de evidência (ainda pendente desde 08-17).
- Considerar resolver o resumo "com perda" de conversas antigas, se o perfil de aprendizado entre sessões passar a importar.

---

## 📅 Data
- **2026-08-31**

### 📌 Status do Projeto
Rodada de melhorias na interface do app do aluno (Chainlit). Nenhuma mudança no grafo/pipeline RAG; o que mudou foi a camada de UI e o empacotamento da KB.

**1. Tela de login — nome deixa de aparecer como senha.** O campo "Nome completo" é o campo de senha nativo do Chainlit realocado (ver entrada de 08-24), então vinha mascarado como pontinhos. Chainlit já renderiza um botão de mostrar/ocultar ao lado do campo; `public/login-unmask.js` (carregado via `custom_js` em `.chainlit/config.toml`) só clica nesse toggle uma vez ao abrir a tela de login, deixando o nome visível como texto por padrão. Usa `MutationObserver` porque o formulário monta de forma assíncrona (SPA React); depende do DOM do login do Chainlit 2.11 (input `#password` com um `<button>` irmão), então um upgrade do Chainlit pode exigir ajuste no seletor.

**2. Pop-up ao clicar "Novo chat" removido.** `confirm_new_chat = false` em `.chainlit/config.toml`. O aviso ("isto vai apagar o histórico do chat atual") é o texto genérico do Chainlit para o caso sem persistência — mas o app tem data layer, então "Novo chat" só abre uma thread nova e as anteriores continuam na sidebar. O aviso era enganoso.

**3. Página "Leia-me" gerada por container.** Antes era um `chainlit.md` estático genérico. Agora:
- `rag/knowledge_base.describe_course_materials(nome)` lê os metadados dos chunks direto do vector store persistido (via cliente `chromadb`, sem embeddings nem chamada de modelo) e devolve a lista de materiais indexados (arquivo + capítulo + título de capítulo quando detectado).
- `chainlit_app._render_readme()` monta o markdown a partir do nome da disciplina + essa lista, e escreve em `chainlit_pt-BR.md` no import do módulo. Chainlit relê esse arquivo a cada request de `/project/settings`, e o idioma está fixo em `pt-BR`, então o arquivo gerado sempre vence.
- `chainlit.md` virou o fallback estático (usado só se a geração falhar). `chainlit_pt-BR.md` entrou no `.gitignore` (gerado por container, efêmero).
- Nada hardcoded sobre a disciplina — tudo sai da KB embutida na imagem.

**4. Citação no chat vira link para o PDF da fonte, na página certa.** Maior mudança da rodada.
- **Empacotamento:** `create_and_save_knowledge_base` agora copia os PDFs de origem para `data/knowledge_bases/<id>/sources/`. Como a árvore inteira da KB já é copiada para a imagem do curso, os arquivos viajam junto sem nenhum passo extra do professor. `pdfs/` (pasta de trabalho) continua fora da imagem — comentário em `.dockerignore` atualizado. Migração manual feita para a KB existente (`kb_566465a2`): os 4 PDFs de `pdfs/` copiados para `sources/`. KBs criadas antes dessa mudança precisam dessa cópia manual.
- `rag/knowledge_base`: novos `knowledge_base_dir(nome)` e `resolve_source_pdf(nome, arquivo)`.
- `chainlit_app._linkify_citations(final_state, answer)`: para cada citação presente na resposta cujo PDF de origem existe em `sources/`, substitui a string longa da citação (`[arquivo, Chapter X, Section Y – ..., Pages Z]`) por um rótulo curto `📄 <arquivo>, p. N` (ou `pp. N–M`) e cria um elemento `cl.Pdf(display="side", page=page_start)`. O Chainlit transforma qualquer elemento cujo `name` apareça no texto da mensagem num "chip" clicável — então o rótulo curto abre o PDF no painel lateral já na página citada. Um elemento por par `(arquivo, página)`; citações sem PDF salvo mantêm o texto completo + badge de código (`highlight_citations`, inalterado).
- O mapa citação→metadados usa `final_state["retrieved_docs"]` + `final_state["evidence"]` (alinhados por índice — `extract_evidence` faz `zip` na mesma ordem).
- `public/style.css` (via `custom_css`): tira o `text-transform: uppercase` e o `0.7rem` do `.element-link` do Chainlit pra o chip ficar legível.
- Verificado no bundle do frontend: o viewer de PDF lateral respeita o prop `page` como página inicial (com clamp à faixa válida).

### ⚠️ Limitações/trade-offs desta rodada
- A citação inline perdeu o detalhe verboso (capítulo/seção/título de seção) — agora isso fica a um clique de distância, dentro do PDF. Foi decisão consciente (o texto longo era a reclamação).
- Quando dois chunks citados caem no mesmo `(arquivo, página)`, compartilham um chip só; o rótulo (e a faixa de páginas) vem do primeiro, então um chunk citado como "p. 2" pode aparecer sob um chip "pp. 2–3".
- Threads antigas (criadas antes desta rodada) podem dar 404 no link do PDF: sem storage de blobs configurado, o arquivo do elemento vive só durante a sessão. Threads novas funcionam. Aceitável para o piloto.
- `chainlit_pt-BR.md` é reescrito no diretório de trabalho a cada boot do container (efêmero, regenerado sempre).
- `login-unmask.js` e o tweak de `.element-link` dependem de detalhes internos do Chainlit 2.11 (DOM do login, classes do frontend) — frágeis a upgrade.

### 🧹 Diversos
- `public/` criado (primeiro uso de assets custom): `login-unmask.js`, `style.css`. Dockerfile passou a `COPY public/ public/`.
- `utils/citations.py` ficou intocado no fim (uma tentativa de parâmetro `skip` foi revertida — a reescrita do texto em `_linkify_citations` tornou desnecessário).

### 🔜 Próximos Passos (revisão)
- Mecanismo de avaliação da qualidade pedagógica da resposta (ainda pendente desde 08-17).
- Atualizar `main.py` para o padrão multi-KB (ainda pendente desde 08-17).
- Paralelizar/batchar grading e extração de evidência (ainda pendente desde 08-17).
- Considerar resolver o resumo "com perda" de conversas antigas, se o perfil de aprendizado entre sessões passar a importar.
- Se persistência de elementos entre sessões passar a importar: configurar um storage client pro data layer (hoje os PDFs anexados só valem na sessão).

---

## 📅 Data
- **2026-08-31** (métricas pedagógicas)

### 📌 Status do Projeto
O grafo do agente já calcula sinais pedagógicos ricos por turno (`LearningState`, `AnswerPlan`, `TeachingState`, `evidence` — ver `agent/state.py`), mas eles eram descartados depois de cada resposta. Agora são capturados, de forma **anônima**, para o professor olhar sob demanda.

**Modelo de dados.** Novo SQLite dedicado, `data/chats/metrics.db` — separado do `chainlit.db` (schema do framework, muda de versão pra versão), no mesmo volume já montado (`/app/data/chats` no Railway). Uma tabela `turn_metrics`, uma linha por turno: `id` (uuid aleatório na escrita), `createdAt`, `discipline`, `topic`/`subtopic`/`intent`/`comprehensionLevel`/`learningProgress`/`frustrationLevel`/`currentDifficulty` (do `LearningState`), `strategy`/`responseDepth` (do `AnswerPlan`), `teachingMode`/`teachingStage` (do `TeachingState`), `studentProfile`, e `citations` (JSON: `[{file, section_id, section_title, page_start, citation}]` derivado de `zip(retrieved_docs, evidence)`).

**Anonimização por omissão, não por hash.** Nada de matrícula, `thread_id`/usuário do Chainlit, nem nada que ligue a linha a um aluno. Cada linha é um evento independente — sem correlação entre turnos da mesma sessão. Dá pra ver "a turma travou no tópico X", não "o aluno Y evoluiu no semestre". Se um dia precisar de granularidade por sessão sem reidentificar, dá pra evoluir com um token efêmero gerado no `on_chat_start` e guardado só em `cl.user_session` — mas não é o ponto de partida.

**Captura.** Novo módulo `utils/metrics.py` (`ensure_metrics_schema`, `record_turn`), só stdlib — não adiciona dependência ao runtime deployado. `ensure_metrics_schema()` roda no nível de módulo em `chainlit_app.py` (junto do `_ensure_sqlite_schema()` que já existia). `record_turn(final_state, DISCIPLINE)` é chamado dentro de `@cl.on_message`, logo depois de `cl.user_session.set("state", final_state)`, via `asyncio.to_thread` (escrita SQLite síncrona, mesma lição do `_run_graph_sync` — não travar o event loop compartilhado). Extração defensiva (`_field` tolera model/dict/None — o primeiro turno pode não ter tudo populado) e a escrita inteira embrulhada em `try/except` que loga warning e segue, igual ao `execute_sql` do `SQLAlchemyDataLayer` do Chainlit.

**Painel do professor.** `app.py` deixou de ser página única — agora tem `st.tabs(["Criar disciplina", "Métricas da turma"])`. A aba de métricas pede o caminho do `metrics.db` (default `data/chats/metrics.db`) ou aceita upload, lê pra um DataFrame e mostra agregados com componentes nativos do Streamlit: turnos por tópico, distribuição de compreensão e de progresso (destaque pra "travado"), frustração média por dia (linha), estratégias mais usadas, e **cobertura do material** — cruza as `citations` registradas com `rag.knowledge_base.list_material_sections()` (lê os metadados dos chunks direto do Chroma) pra listar as seções que nunca apareceram numa resposta. Sem autenticação — `app.py` não é deployado pros alunos.

**Puxar o arquivo.** Documentado em `DEPLOY.md` (seção "Class metrics"): `railway volume files download /metrics.db ./data/chats/metrics.db --overwrite` (sintaxe confirmada contra o CLI 5.43.1 — o path é relativo à raiz do volume, que é o mount point), ou `docker cp` no caso Docker puro.

**Refactor de tabela.** `rag/knowledge_base.py`: a leitura de metadados do Chroma que estava dentro de `describe_course_materials` virou `_kb_chunk_metadatas()`, reusada por `describe_course_materials` e pelo novo `list_material_sections`.

### ⚠️ Limitações/trade-offs desta rodada
- **`StudentProfile` é inerte hoje.** Está definido em `agent/state.py` e passa pelo `TutorState`, mas nenhum nó do grafo o atualiza — a coluna `studentProfile` vai ser sempre `"neutral"`. Gravada mesmo assim (barata, à prova de futuro), mas não espere sinal aí até o grafo populá-la.
- Sem correlação entre turnos: nenhuma visão de "trajetória" de aluno ou de sessão. Foi a troca aceita.
- `citations` registra toda a evidência que passou pelo grading do turno, não só o que efetivamente foi citado no texto final — é uma leve superestimativa de "o que a turma consultou", ok pra análise de cobertura.
- Retenção: pra um piloto de um semestre, deixar crescer sem limite deve bastar. Revisar se virar multi-semestre.
- `pandas` foi fixado explicitamente em `requirements.txt` (só usado pelo `app.py` local; já vinha como dependência transitiva do Streamlit, mas o freeze não capturava). `use_container_width` (depreciado) trocado por `width="stretch"` no `app.py`.

### 🔜 Próximos Passos (revisão)
- Fazer algum nó do grafo de fato popular o `StudentProfile` (ou remover o campo se não for pra usar).
- Mecanismo de avaliação da qualidade pedagógica da resposta (ainda pendente desde 08-17).
- Atualizar `main.py` para o padrão multi-KB (ainda pendente desde 08-17).
- Paralelizar/batchar grading e extração de evidência (ainda pendente desde 08-17).
- Depois de um uso real em produção: baixar o `metrics.db` via `railway volume files download`, abrir no `app.py` e confirmar que os agregados batem com o que rolou no chat.