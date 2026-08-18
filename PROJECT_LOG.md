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