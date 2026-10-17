from agents.literature_agent import LiteratureAgent
from agents.clinical_agent import ClinicalAgent
from agents.critic_agent import CriticAgent
from agents.synthesizer_agent import SynthesizerAgent


class SimpleECGOrchestrator:

    def __init__(self):

        self.literature_agent = LiteratureAgent()
        self.clinical_agent = ClinicalAgent()
        self.critic_agent = CriticAgent()
        self.synthesizer_agent = SynthesizerAgent()

        self.max_iterations = 1

    def run(self, question):

        print("\n[1] LiteratureAgent")

        literature = self.literature_agent.run(question)

        context = literature["context"]

        print("[2] ClinicalAgent")

        clinical = self.clinical_agent.run(
            question,
            context,
        )

        clinical_answer = clinical["answer"]

        print("[3] CriticAgent")

        critic = self.critic_agent.run(
            question,
            context,
            clinical_answer,
        )

        print("Critic verdict :", critic["verdict"])
        print("Critic score   :", critic.get("score"))

        iteration = 0

        while (
            critic["verdict"] in {"IMPROVE", "FAIL"}
            and iteration < self.max_iterations
        ):

            print("[4] Correction ClinicalAgent")

            feedback = "\n".join(
                critic.get("feedback", [])
            )

            clinical = self.clinical_agent.run(
                question,
                context,
                critic_feedback=feedback,
            )

            clinical_answer = clinical["answer"]

            critic = self.critic_agent.run(
                question,
                context,
                clinical_answer,
            )

            iteration += 1

        print("[5] SynthesizerAgent")

        synthesis = self.synthesizer_agent.run(
            question,
            clinical_answer,
            critic,
        )

        return {
            "question": question,
            "literature": literature,
            "clinical": clinical,
            "critic": critic,
            "final_answer": synthesis["answer"],
            "iterations": iteration,
        }


if __name__ == "__main__":

    orchestrator = SimpleECGOrchestrator()

    question = (
        "Quels sont les critères ECG diagnostiques "
        "de la fibrillation atriale selon les "
        "recommandations ESC 2024 ?"
    )

    result = orchestrator.run(question)

    print("\n" + "=" * 80)
    print("RÉPONSE FINALE")
    print("=" * 80)

    print(result["final_answer"])