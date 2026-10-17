"""Tests de evaluation/recalculate_metrics.py (version corrigée).

Lancer :  python -m pytest evaluation/tests/test_recalculate_metrics.py -v
(ou :     python -m unittest evaluation.tests.test_recalculate_metrics -v)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier evaluation/
import recalculate_metrics as m


class ExtractDirectAnswerTests(unittest.TestCase):
    def test_plain_heading(self):
        text = "1. Réponse directe\nLe texte utile.\n\n2. Critères\nAutre chose."
        self.assertEqual(m.extract_direct_answer(text), "Le texte utile.")

    def test_markdown_bold_heading(self):
        text = "**1. Réponse directe**  \nLe texte utile.\n\n2. Critères\nAutre chose."
        self.assertEqual(m.extract_direct_answer(text), "Le texte utile.")

    def test_stops_before_ptbxl_traceability_block(self):
        text = "1. Réponse directe\nTexte clinique.\n\nTraçabilité PTB-XL\nCas PTB-XL : 123"
        self.assertEqual(m.extract_direct_answer(text), "Texte clinique.")

    def test_section_running_to_end_of_text(self):
        text = "1. Réponse directe\nSeul contenu, sans autre section."
        self.assertEqual(m.extract_direct_answer(text), "Seul contenu, sans autre section.")

    def test_missing_heading_returns_none(self):
        text = "La réponse clinique générée n'a pas passé la vérification finale."
        self.assertIsNone(m.extract_direct_answer(text))

    def test_empty_input_returns_none(self):
        self.assertIsNone(m.extract_direct_answer(""))
        self.assertIsNone(m.extract_direct_answer(None))


class TokenizeNormalizationTests(unittest.TestCase):
    def test_non_breaking_hyphen_matches_ascii_hyphen(self):
        """Bug corrigé : le LLM sort U+2011, reference_answer utilise '-' ASCII."""
        with_special_hyphen = m.tokenize("auriculo\u2011ventriculaire")
        with_ascii_hyphen = m.tokenize("auriculo-ventriculaire")
        self.assertEqual(with_special_hyphen, with_ascii_hyphen)
        self.assertEqual(with_special_hyphen, ["auriculo-ventriculaire"])

    def test_narrow_no_break_space_becomes_word_boundary(self):
        # espace fine insécable (U+202F) entre des ID de cas PTB-XL : le mot
        # « et » doit être isolé, pas fusionné avec les nombres voisins.
        tokens = m.tokenize("1799,\u202f1075\u202fet\u202f1815")
        self.assertNotIn("1799,\u202f1075", tokens)
        self.assertEqual(tokens, ["et"])   # les identifiants numériques purs sont exclus (cf. test suivant)

    def test_curly_apostrophe_normalized(self):
        self.assertEqual(m.tokenize("n\u2019est"), m.tokenize("n'est"))

    def test_trailing_punctuation_does_not_attach_to_word(self):
        self.assertIn("préservée", m.tokenize("préservée."))
        self.assertNotIn("préservée.", m.tokenize("préservée."))

    def test_pure_digits_are_dropped(self):
        # évite qu'une suite de numéros de page ("702 703 704...") gonfle les scores
        self.assertEqual(m.tokenize("702 703 704"), [])


class RougeBleuMeteorTests(unittest.TestCase):
    def test_identical_texts_score_near_one(self):
        text = "Le diagnostic repose sur des ondes P absentes et des intervalles RR irréguliers."
        rouge = m.calculate_rouge(text, text)
        self.assertGreater(rouge["rouge1"], 0.99)
        self.assertGreater(m.calculate_bleu(text, text), 0.9)

    def test_unrelated_texts_score_low(self):
        ref = "Le diagnostic de fibrillation atriale requiert une documentation ECG."
        pred = "Le chat mange une souris dans le jardin ensoleillé."
        rouge = m.calculate_rouge(ref, pred)
        self.assertLess(rouge["rouge1"], 0.3)

    def test_bleu_and_meteor_none_on_empty_text(self):
        self.assertIsNone(m.calculate_bleu("", "quelque chose"))
        self.assertIsNone(m.calculate_bleu("quelque chose", ""))

    def test_meteor_rewards_synonyms_more_than_bleu(self):
        # cas typique de reformulation clinique légitime
        ref = "absence d'ondes P répétitives discernables"
        pred = "absence d'ondes P répétées et discernables"
        meteor = m.calculate_meteor(ref, pred)
        self.assertIsNotNone(meteor)
        self.assertGreater(meteor, 0.5)


class BertscoreBatchTests(unittest.TestCase):
    def test_empty_input_returns_empty_list(self):
        self.assertEqual(m.calculate_bertscore_batch([], []), [])

    def test_network_failure_returns_none_without_raising(self):
        with patch("bert_score.score", side_effect=OSError("no network")):
            result = m.calculate_bertscore_batch(["ref"], ["pred"])
        self.assertIsNone(result)

    def test_success_path_returns_one_float_per_row(self):
        import torch

        fake_f1 = torch.tensor([0.8, 0.6])
        with patch("bert_score.score", return_value=(None, None, fake_f1)):
            result = m.calculate_bertscore_batch(["r1", "r2"], ["p1", "p2"])
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 0.8, places=5)
        self.assertAlmostEqual(result[1], 0.6, places=5)


class ScoreResultsIntegrationTests(unittest.TestCase):
    """Teste score_results() de bout en bout, sans toucher au disque ni au réseau."""

    def setUp(self):
        self.questions = [
            {"id": 1, "reference_answer": "Absence d'ondes P et intervalles RR irréguliers."},
            {"id": 2, "reference_answer": "Texte de référence pour la question 2."},
            {"id": 3, "reference_answer": ""},                    # référence manquante
            {"id": 4, "reference_answer": "Une référence pour la question 4."},
        ]
        self.results = [
            {
                "id": 1,
                "publication_status": "PASS",
                "final_answer": "1. Réponse directe\nAbsence d'ondes P et intervalles RR irréguliers.\n\n"
                                "2. Critères\nBeaucoup de détails supplémentaires non pertinents ici.",
            },
            {
                "id": 2,
                "publication_status": "REJECTED",
                "final_answer": "La réponse clinique générée n'a pas passé la vérification finale.",
            },
            {
                "id": 3,
                "publication_status": "PASS",
                "final_answer": "1. Réponse directe\nContenu quelconque.",
            },
            {
                "id": 4,
                "publication_status": "PASS",
                "final_answer": "Format inattendu, sans section numérotée.",
            },
        ]

    def test_skip_reasons_are_assigned_correctly(self):
        with patch.object(m, "calculate_bertscore_batch", return_value=None):
            m.score_results(self.results, self.questions)

        reasons = {row["id"]: row["metrics_skipped_reason"] for row in self.results}
        self.assertIsNone(reasons[1])
        self.assertEqual(reasons[2], "rejected_publication")
        self.assertEqual(reasons[3], "missing_reference")
        self.assertEqual(reasons[4], "no_extractable_section")

    def test_only_direct_answer_section_is_compared(self):
        with patch.object(m, "calculate_bertscore_batch", return_value=None):
            m.score_results(self.results, self.questions)

        row1 = next(r for r in self.results if r["id"] == 1)
        self.assertEqual(row1["metrics_reference_excerpt"],
                         "Absence d'ondes P et intervalles RR irréguliers.")
        self.assertGreater(row1["rouge1"], 0.9)     # section extraite == référence, quasi parfait

    def test_summary_counts_and_means_ignore_skipped_rows(self):
        with patch.object(m, "calculate_bertscore_batch", return_value=None):
            summary = m.score_results(self.results, self.questions)

        self.assertEqual(summary["n_total"], 4)
        self.assertEqual(summary["n_scored"], 1)     # seule la question 1 est scorée
        self.assertEqual(summary["n_skipped"]["rejected_publication"], 1)
        self.assertEqual(summary["n_skipped"]["missing_reference"], 1)
        self.assertEqual(summary["n_skipped"]["no_extractable_section"], 1)
        self.assertIsNotNone(summary["mean_rouge1"])
        self.assertEqual(summary["mean_bertscore"], None)   # bertscore désactivé dans ce test

    def test_bertscore_results_are_attached_per_row(self):
        with patch.object(m, "calculate_bertscore_batch", return_value=[0.77]):
            m.score_results(self.results, self.questions)

        row1 = next(r for r in self.results if r["id"] == 1)
        self.assertEqual(row1["bertscore"], 0.77)

    def test_one_row_metric_failure_does_not_stop_the_others(self):
        """Une métrique qui explose sur une ligne ne doit pas empêcher les autres lignes."""
        with patch.object(m, "calculate_rouge", side_effect=[RuntimeError("boom"), {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}]), \
             patch.object(m, "calculate_bertscore_batch", return_value=None):
            extra_question = {"id": 5, "reference_answer": "Une autre référence valable."}
            extra_result = {
                "id": 5, "publication_status": "PASS",
                "final_answer": "1. Réponse directe\nUne autre référence valable.",
            }
            questions = [self.questions[0], extra_question]
            results = [self.results[0], extra_result]
            summary = m.score_results(results, questions)

        self.assertNotIn("rouge1", results[0])          # ligne 1 : échec, pas de clé posée
        self.assertIn("rouge1", results[1])              # ligne 5 : calculée normalement
        self.assertEqual(summary["n_failed_metrics"].get("rouge"), 1)
        self.assertEqual(summary["n_scored"], 2)          # les deux passent quand même en bertscore


if __name__ == "__main__":
    unittest.main()