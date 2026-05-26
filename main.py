from rag.splitter import spliting_documents
from rag.vectorstore import build_vectorstore
from agent.tools import build_retrieve_tool
import agent.prompts as prompts
from agent.graph import build_graph
from agent.state import StudentProfile
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from agent.state import TutorState, TutorConfig

load_dotenv()

from pathlib import Path
from rag.loader import load_documents, parse_documents

if __name__ == "__main__":
  
  docs = load_documents(Path("pdfs/"))

  parsed_docs = parse_documents(docs)

  chunks_table = spliting_documents(parsed_docs)
  
  retriever = build_vectorstore(chunks_table)

  retrieve_tool = build_retrieve_tool(retriever)

  config = TutorConfig(
    subject="Software Architecture"
  )

  tutor_prompt = SystemMessage(content=prompts.SYSTEM_PROMPT.format(
        domain=config.subject,
        max_sentences=config.max_sentences,
        course_level=config.course_level,
        answer_language=config.answer_language
        )
    )

  response_model = init_chat_model(
      os.getenv("MODEL_NAME", "gpt-4o-mini"), 
      temperature=os.getenv("MODEL_TEMPERATURE", 0)
  ).bind_tools([retrieve_tool])

  graph = build_graph(config, retrieve_tool, response_model)

  print("\n===== RAG INTERATIVO (COM ETAPAS) =====")
  print("Digite sua pergunta ou 'sair' para encerrar.\n")

  conversation_state = TutorState(
    messages=[tutor_prompt],
    student_profile=StudentProfile(),
    current_topic=None
  )

  print("Student Profile:", conversation_state["student_profile"])

  while True:
      question = input("Pergunta: ").strip()

      if question.lower() in ["sair", "exit", "quit"]:
          print("Encerrando...")
          break

      print("\n===== INÍCIO DO PIPELINE RAG =====\n")

      conversation_state["messages"].append(
          HumanMessage(content=question)
      )

      for step in graph.stream(conversation_state): # type: ignore
          for node_name, state in step.items():
              print(f"\n--- NÓ EXECUTADO: {node_name} ---")

              # DEBUG DO PERFIL
              if "student_profile" in state:
                  profile = state["student_profile"]

                  print("\n[DEBUG] Perfil do aluno:")
                  print(f"perfil_atual: {profile.current_profile}")
                  print(f"confianca: {profile.confidence:.2f}")
                  print("sinais:")
                  print(f"  pede_exercicio: {profile.asks_exercise}")
                  print(f"  pede_detalhe: {profile.asks_detail}")
                  print(f"  pede_objetividade: {profile.asks_objectivity}")

              if "messages" in state:
                  print("\nMensagens:")
                  for msg in state["messages"]:
                      print(f"\nTipo: {type(msg).__name__}")

                      if isinstance(msg, HumanMessage):
                          print("Human:", msg.content)

                      elif isinstance(msg, AIMessage):
                          if msg.content:
                              print("AI:", msg.content)
                          if msg.tool_calls:
                              print("Tool calls:", msg.tool_calls)

                      elif isinstance(msg, ToolMessage):
                          print("Tool result:", msg.content)

      conversation_state = state

      print("\n===== RESPOSTA FINAL =====\n")
      print(conversation_state["messages"][-1].content)
      print("\n===== PERFIL DO ALUNO =====\n")
      print(f"Perfil atual: {conversation_state['student_profile'].current_profile}")
      print(f"Confiança: {conversation_state['student_profile'].confidence:.2f}")
      print("\n==========================\n")