import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from agent.chat_pipeline import load_pipeline
from agent.state import StudentProfile, TutorState

if __name__ == "__main__":

    discipline = os.getenv("DISCIPLINE", "").strip()
    if not discipline:
        sys.exit(
            "Defina DISCIPLINE com o nome de uma disciplina em "
            "data/knowledge_bases/ (o nome dado ao criar a base), por exemplo:\n"
            '  DISCIPLINE="Estruturas de Dados Básicas II" python main.py'
        )

    graph, tutor_prompt, config = load_pipeline(discipline)

    print("\n===== RAG INTERATIVO (COM ETAPAS) =====")
    print(f"Disciplina: {discipline}")
    print("Digite sua pergunta ou 'sair' para encerrar.\n")

    conversation_state = TutorState(
        messages=[tutor_prompt],
        student_profile=StudentProfile(),
    )  # type: ignore

    while True:
        question = input("Pergunta: ").strip()

        if question.lower() in ["sair", "exit", "quit"]:
            print("Encerrando...")
            break

        conversation_state["messages"].append(
            HumanMessage(content=question)
        )

        final_state = None

        try:
            for state in graph.stream(conversation_state, stream_mode="values"):  # type: ignore
                final_state = state
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            print(f"\n[erro ao processar a pergunta] {exc}\n")
            conversation_state["messages"].pop()
            continue

        conversation_state = final_state

        print("\n===== RESPOSTA FINAL =====\n")
        print(conversation_state["messages"][-1].content)
        print("\n==========================\n")
