import os

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from agent.chat_pipeline import load_pipeline
from agent.state import StudentProfile, TutorState

if __name__ == "__main__":

    discipline = os.getenv("DISCIPLINE", "Software Architecture")

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

        for state in graph.stream(conversation_state, stream_mode="values"):  # type: ignore
            final_state = state

        conversation_state = final_state

        print("\n===== RESPOSTA FINAL =====\n")
        print(conversation_state["messages"][-1].content)
        print("\n==========================\n")
