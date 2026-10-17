import sys
from pathlib import Path
from typing import Dict, Any
import os

class SignalAgent:
    """
    Agent chargé d'analyser une entrée ECG.

    Il réutilise le projet ECG existant :
        image
        -> extraction
        -> digitisation
        -> classification

    Il ne pose pas de diagnostic médical.
    Il retourne uniquement les résultats du modèle.
    """

    @staticmethod
    def separate_class_decisions(classification: dict):
        """Sépare les classes au-dessus du seuil de la classe de repli."""

        raw_classes = classification.get("predicted_classes") or []
        details = classification.get("details")
        if not isinstance(details, list) or not details:
            return [], None, "unverified"

        threshold_classes = [
            detail["class"]
            for detail in details
            if isinstance(detail, dict)
            and detail.get("predicted") is True
            and isinstance(detail.get("class"), str)
        ]
        if threshold_classes:
            return threshold_classes, None, "threshold"

        if len(raw_classes) == 1:
            selected_class = raw_classes[0]
            detail = next(
                (
                    item for item in details
                    if isinstance(item, dict)
                    and item.get("class") == selected_class
                    and item.get("predicted") is False
                ),
                None,
            )
            if detail is not None:
                try:
                    probabilities = classification.get("probabilities") or {}
                    score = float(probabilities[selected_class])
                    threshold = float(detail["threshold"])
                    highest_score = max(
                        float(value) for value in probabilities.values()
                    )
                except (KeyError, TypeError, ValueError):
                    return [], None, "unverified"
                if score < threshold and score == highest_score:
                    return [], selected_class, "argmax_fallback"

        return [], None, "none" if not raw_classes else "unverified"

    def __init__(self, ecg_project_path: str | None = None):

        # Résolution robuste du chemin du projet ECG.
        import os

        candidates = []

        env_path = os.getenv("ECG_PROJECT_PATH")
        if env_path:
            candidates.append(Path(env_path))

        if ecg_project_path:
            candidates.append(Path(ecg_project_path))

        # Chemin réel attendu dans ton installation.
        candidates.append(Path(r"D:\ecg_agent_project\ecg_project"))

        selected_path = None

        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except Exception:
                resolved = candidate

            # On vérifie aussi la présence du module attendu.
            if resolved.exists() and (resolved / "ecg_agent.py").exists():
                selected_path = resolved
                break

        if selected_path is None:
            tested = "\n".join(
                f"- {candidate}" for candidate in candidates
            )
            raise FileNotFoundError(
                "Projet ECG introuvable. Chemins testés :\n"
                f"{tested}\n\n"
                "Le dossier doit contenir le fichier ecg_agent.py."
            )

        self.ecg_project_path = selected_path

        # Ajouter le projet ECG au PYTHONPATH
        if str(self.ecg_project_path) not in sys.path:
            sys.path.insert(
                0,
                str(self.ecg_project_path)
            )

        # Import après ajout du chemin
        from ecg_agent import ecg_agent

        self.ecg_agent = ecg_agent

    def run(
        self,
        ecg_input_path: str | None
    ) -> Dict[str, Any]:

        # Aucun ECG fourni
        if not ecg_input_path:
            return {
                "agent": "SignalAgent",
                "status": "no_signal",
                "input_path": None,
                "signal_path": None,
                "predicted_classes": [],
                "raw_predicted_classes": [],
                "fallback_class": None,
                "selection_method": "none",
                "probabilities": {},
                "model": None,
                "details": [],
            }

        input_path = Path(
            ecg_input_path
        ).resolve()

        if not input_path.exists():
            return {
                "agent": "SignalAgent",
                "status": "error",
                "error": (
                    f"Fichier ECG introuvable : "
                    f"{input_path}"
                ),
            }
        print("=== SIGNAL DEBUG ===")
        print("Python:", sys.executable)
        print("CWD:", os.getcwd())
        print("ECG project:", self.ecg_project_path)
        print("Input:", input_path)
        print("====================")
        try:
            # Appel du pipeline étudiant 1
            result = self.ecg_agent(
                str(input_path)
            )

            classification = result.get(
                "classification_result",
                {},
            )
            threshold_classes, fallback_class, selection_method = (
                self.separate_class_decisions(classification)
            )
            if selection_method in {"none", "unverified"}:
                raise ValueError(
                    "Sortie du classifieur incomplète ou incohérente : "
                    "impossible de vérifier la décision par seuil ou par repli."
                )

            return {
                "agent": "SignalAgent",

                "status": "success",

                "input_path": str(
                    input_path
                ),

                "signal_path": result.get(
                    "signal_path_used_by_classifier"
                ),

                "model": classification.get(
                    "model"
                ),

                "predicted_classes": threshold_classes,
                "classes_above_threshold": threshold_classes,
                "raw_predicted_classes": classification.get(
                    "predicted_classes", []
                ),
                "fallback_class": fallback_class,
                "selection_method": selection_method,
                "decision_mode": selection_method,
                "input_type": "image",

                "probabilities": (
                    classification.get(
                        "probabilities",
                        {},
                    )
                ),

                "details": classification.get(
                    "details",
                    [],
                ),

                "n_windows_used": (
                    classification.get(
                        "n_windows_used"
                    )
                ),

                "device": classification.get(
                    "device"
                ),
            }

        except Exception as exc:

            return {
                "agent": "SignalAgent",
                "status": "error",
                "input_path": str(
                    input_path
                ),
                "error": str(exc),
            }
