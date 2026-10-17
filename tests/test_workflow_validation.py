"""Tests sans appel réseau pour la validation de la réponse clinique."""

import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents.critic_agent import CriticAgent
from agents.signal_agent import SignalAgent


LIMITS = "4. Limites du contexte\nLes extraits ne précisent rien de plus.\n\n5. Avertissement médical\nAvis médical."
FORBIDDEN_LIMITS = (
    "4. Limites du contexte\nLes extraits ne précisent pas, par ex., ce point.\n\n"
    "5. Avertissement médical\nAvis médical."
)
FALLBACK_SIGNAL = {
    "status": "success",
    "predicted_classes": ["NORM"],
    "probabilities": {"NORM": 0.8840736746788025},
    "details": [
        {"class": "NORM", "threshold": 0.90, "predicted": False}
    ],
}


def passing_critic():
    critic = CriticAgent.__new__(CriticAgent)
    criteria = {name: "PASS" for name in CriticAgent.ALL_CRITERIA}
    critic.llm = Mock()
    critic.llm.invoke.return_value = SimpleNamespace(
        content=json.dumps({"criteria": criteria, "feedback": []})
    )
    return critic


class CriticValidationTests(unittest.TestCase):
    def test_forbidden_variants_only_in_limits(self):
        for expression in ("par exemple", "par ex.", "ex."):
            with self.subTest(expression=expression):
                answer = LIMITS.replace("rien de plus", expression)
                self.assertTrue(CriticAgent.forbidden_example_in_limits(answer))
        self.assertFalse(
            CriticAgent.forbidden_example_in_limits(
                "3. Justification documentaire\nVoir par ex. le passage 2.\n" + LIMITS
            )
        )
        self.assertTrue(
            CriticAgent.forbidden_example_in_limits(
                FORBIDDEN_LIMITS.replace(
                    "4. Limites du contexte", "4. Limites du contexte\\"
                )
            )
        )

    def test_citation_style_failure_reduces_score(self):
        result = passing_critic().run("Question", "Contexte", FORBIDDEN_LIMITS)
        self.assertEqual(result["criteria"]["citation_style"], "FAIL")
        self.assertEqual((result["score"], result["max_score"]), (6, 7))
        self.assertEqual(result["verdict"], "IMPROVE")
        self.assertTrue(result["feedback"])


class SignalDecisionTests(unittest.TestCase):
    def test_fallback_is_not_reported_as_threshold_prediction(self):
        agent = SignalAgent.__new__(SignalAgent)
        agent.ecg_agent = Mock(return_value={
            "classification_result": {
                "predicted_classes": ["NORM"],
                "probabilities": {"NORM": 0.8840736746788025},
                "details": [
                    {"class": "NORM", "threshold": 0.90, "predicted": False}
                ],
            }
        })
        with patch("agents.signal_agent.Path.exists", return_value=True):
            result = agent.run("ecg.png")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["predicted_classes"], [])
        self.assertEqual(result["raw_predicted_classes"], ["NORM"])
        self.assertEqual(result["fallback_class"], "NORM")
        self.assertEqual(result["selection_method"], "argmax_fallback")

    def test_threshold_prediction_keeps_predicted_class(self):
        classes, fallback, method = SignalAgent.separate_class_decisions({
            "predicted_classes": ["NORM"],
            "details": [
                {"class": "NORM", "threshold": 0.90, "predicted": True}
            ],
        })
        self.assertEqual(classes, ["NORM"])
        self.assertIsNone(fallback)
        self.assertEqual(method, "threshold")

    def test_missing_details_does_not_claim_threshold_prediction(self):
        classes, fallback, method = SignalAgent.separate_class_decisions({
            "predicted_classes": ["NORM"],
            "probabilities": {"NORM": 0.884},
        })
        self.assertEqual(classes, [])
        self.assertIsNone(fallback)
        self.assertEqual(method, "unverified")

    def test_incomplete_classifier_output_is_an_error(self):
        agent = SignalAgent.__new__(SignalAgent)
        agent.ecg_agent = Mock(return_value={
            "classification_result": {
                "predicted_classes": ["NORM"],
                "probabilities": {"NORM": 0.884},
            }
        })
        with patch("agents.signal_agent.Path.exists", return_value=True):
            result = agent.run("ecg.png")
        self.assertEqual(result["status"], "error")
        self.assertIn("incomplète ou incohérente", result["error"])


class WorkflowPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Les agents du module sont instanciés à l'import : neutraliser leurs clients.
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("agents.groq_client.ChatGroq"):
                import orchestrator.langgraph_orchestrator as workflow
        cls.workflow = workflow

    def run_workflow(self, clinical_answers, synthesized_answer):
        workflow = self.workflow
        with patch.object(workflow, "literature_agent") as literature, \
             patch.object(workflow, "clinical_agent") as clinical, \
             patch.object(workflow, "critic_agent", passing_critic()), \
             patch.object(workflow, "synthesizer_agent") as synthesizer:
            literature.run.return_value = {"context": "Contexte de test"}
            clinical.run.side_effect = [
                {"answer": answer} for answer in clinical_answers
            ]
            synthesizer.run.return_value = {"answer": synthesized_answer}
            result = workflow.run_agentic_rag("Question de test")
            return result, clinical.run.call_count

    def test_signal_error_is_printed_and_returned(self):
        signal_agent = Mock()
        signal_agent.run.return_value = {
            "status": "error",
            "error": "Fichier ECG introuvable",
        }
        output = StringIO()
        with patch.object(self.workflow, "get_signal_agent", return_value=signal_agent):
            with redirect_stdout(output):
                result = self.workflow.signal_node({"ecg_input_path": "ecg.png"})
        self.assertEqual(result["signal_result"]["status"], "error")
        self.assertIn("SignalAgent status : error", output.getvalue())
        self.assertIn("ERREUR SIGNALAGENT : Fichier ECG introuvable", output.getvalue())

    def test_signal_error_without_message_uses_fallback(self):
        signal_agent = Mock()
        signal_agent.run.return_value = {"status": "error"}
        output = StringIO()
        with patch.object(self.workflow, "get_signal_agent", return_value=signal_agent):
            with redirect_stdout(output):
                self.workflow.signal_node({"ecg_input_path": "ecg.png"})
        self.assertIn("ERREUR SIGNALAGENT : Erreur inconnue", output.getvalue())

    def test_signal_error_stops_after_join_before_clinical(self):
        signal_agent = Mock()
        signal_agent.run.return_value = {
            "status": "error",
            "error": "Échec de la classification",
        }
        with patch.object(self.workflow, "get_signal_agent", return_value=signal_agent), \
             patch.object(self.workflow, "literature_agent") as literature, \
             patch.object(self.workflow, "clinical_agent") as clinical:
            result = self.workflow.run_agentic_rag(
                "Analyse cet ECG", ecg_input_path="ecg.png", ecg_required=True
            )
        literature.run.assert_called_once()
        clinical.run.assert_not_called()
        self.assertEqual(result["signal_result"]["status"], "error")
        self.assertEqual(result["publication_status"], "REJECTED")
        self.assertEqual(result["final_answer"], self.workflow.SIGNAL_ERROR_ANSWER)
        self.assertEqual(result["final_critic_result"], {})

    def test_missing_required_ecg_stops_after_join(self):
        with patch.object(self.workflow, "literature_agent") as literature:
            result = self.workflow.run_agentic_rag(
                "Analyse cet ECG", ecg_required=True
            )
        literature.run.assert_called_once()
        self.assertEqual(result["signal_result"]["status"], "no_signal")
        self.assertEqual(result["publication_status"], "REJECTED")
        self.assertEqual(result["final_answer"], self.workflow.MISSING_SIGNAL_ANSWER)

    def test_signal_and_literature_run_in_parallel_then_join(self):
        barrier = Barrier(2)
        signal_agent = Mock()
        signal_agent.run.side_effect = lambda path: (
            barrier.wait(timeout=3),
            {"status": "success", "predicted_classes": [], "probabilities": {}},
        )[1]

        def retrieve(question):
            barrier.wait(timeout=3)
            return {"context": "Document RAG disponible"}

        def clinical_answer(question, context, critic_feedback):
            self.assertIn("Document RAG disponible", context)
            self.assertIn("RÉSULTAT SIGNALAGENT", context)
            return {"answer": LIMITS}

        with patch.object(self.workflow, "get_signal_agent", return_value=signal_agent), \
             patch.object(self.workflow, "literature_agent") as literature, \
             patch.object(self.workflow, "clinical_agent") as clinical, \
             patch.object(self.workflow, "critic_agent", passing_critic()), \
             patch.object(self.workflow, "synthesizer_agent") as synthesizer:
            literature.run.side_effect = retrieve
            clinical.run.side_effect = clinical_answer
            synthesizer.run.return_value = {"answer": LIMITS}
            result = self.workflow.run_agentic_rag(
                "Analyse cet ECG", ecg_input_path="ecg.png", ecg_required=True
            )

        signal_agent.run.assert_called_once()
        literature.run.assert_called_once()
        clinical.run.assert_called_once()
        self.assertEqual(result["publication_status"], "PASS")

    def test_retries_then_publishes_validated_answer(self):
        result, calls = self.run_workflow(
            [FORBIDDEN_LIMITS, LIMITS], LIMITS
        )
        self.assertEqual(calls, 2)
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(result["publication_status"], "PASS")
        self.assertEqual(result["final_answer"], LIMITS)
        self.assertEqual(result["final_critic_result"]["verdict"], "PASS")

    def test_rejects_synthesizer_output_that_fails_final_review(self):
        result, calls = self.run_workflow([LIMITS], FORBIDDEN_LIMITS)
        self.assertEqual(calls, 1)
        self.assertEqual(result["publication_status"], "REJECTED")
        self.assertEqual(result["final_critic_result"]["verdict"], "IMPROVE")
        self.assertNotIn("par ex.", result["final_answer"])
        self.assertEqual(result["final_answer"], self.workflow.UNVALIDATED_ANSWER)

    def test_rejects_after_retry_limit_when_final_answer_still_fails(self):
        result, calls = self.run_workflow(
            [FORBIDDEN_LIMITS, FORBIDDEN_LIMITS], FORBIDDEN_LIMITS
        )
        self.assertEqual(calls, 2)
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(result["critic_result"]["verdict"], "IMPROVE")
        self.assertEqual(result["publication_status"], "REJECTED")
        self.assertEqual(result["final_answer"], self.workflow.UNVALIDATED_ANSWER)

    def test_rejects_when_final_critic_result_is_missing(self):
        result = self.workflow.publication_node({"final_answer": LIMITS})
        self.assertEqual(result["publication_status"], "REJECTED")
        self.assertEqual(result["final_answer"], self.workflow.UNVALIDATED_ANSWER)

    def test_fallback_class_is_reported_with_score_and_threshold(self):
        context = self.workflow.format_signal_context(FALLBACK_SIGNAL)
        self.assertIn("classe au score maximal", context)
        self.assertIn("catégorie NORM", context)
        self.assertIn("score associé est de 0,884", context)
        self.assertIn("seuil de 0,900", context)

    def test_normalized_fallback_is_reported_with_score_and_threshold(self):
        signal = {
            **FALLBACK_SIGNAL,
            "predicted_classes": [],
            "raw_predicted_classes": ["NORM"],
            "fallback_class": "NORM",
            "selection_method": "argmax_fallback",
        }
        context = self.workflow.format_signal_context(signal)
        self.assertIn('"predicted_classes": []', context)
        self.assertIn('"fallback_class": "NORM"', context)
        self.assertIn("score associé est de 0,884", context)

    def test_fallback_class_rejects_qualitative_probability(self):
        state = {
            "signal_result": FALLBACK_SIGNAL,
            "final_critic_result": {"verdict": "PASS"},
            "final_answer": (
                "Le modèle a classé l'ECG en NORM avec une probabilité élevée "
                "et un score de 0,884."
            ),
        }
        result = self.workflow.publication_node(state)
        self.assertEqual(result["publication_status"], "REJECTED")
        self.assertTrue(result["publication_feedback"])

    def test_fallback_class_accepts_neutral_score_wording(self):
        state = {
            "signal_result": {
                **FALLBACK_SIGNAL,
                "predicted_classes": [],
                "raw_predicted_classes": ["NORM"],
                "fallback_class": "NORM",
                "selection_method": "argmax_fallback",
            },
            "final_critic_result": {"verdict": "PASS"},
            "final_answer": (
                "Le modèle a classé l'ECG dans la catégorie NORM. "
                "Le score associé à cette classe est de 0,884."
            ),
        }
        result = self.workflow.publication_node(state)
        self.assertEqual(result["publication_status"], "PASS")
        self.assertEqual(result["publication_feedback"], [])

    def test_fallback_class_requires_numeric_score(self):
        state = {
            "signal_result": FALLBACK_SIGNAL,
            "final_critic_result": {"verdict": "PASS"},
            "final_answer": "Le modèle a classé l'ECG dans la catégorie NORM.",
        }
        result = self.workflow.publication_node(state)
        self.assertEqual(result["publication_status"], "REJECTED")
        self.assertIn("0,884", result["publication_feedback"][0])


if __name__ == "__main__":
    unittest.main()
