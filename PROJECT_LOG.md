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