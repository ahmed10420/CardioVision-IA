# CardioVision IA

**CardioVision IA** est un système multi-agent d’aide à la décision clinique appliqué à l’ECG.

Il combine :

- analyse d’image ECG ;
- digitalisation du tracé ;
- classification du signal avec ResNet1D ;
- RAG médical basé sur des guidelines et références ECG ;
- recherche de cas PTB-XL similaires ;
- orchestration multi-agent avec LangGraph ;
- génération, critique et validation du rapport clinique.

> Ce projet est un outil académique d’aide à la décision. Il ne remplace pas l’avis d’un professionnel de santé.

## Pipeline

```text
Cas clinique + ECG
        ↓
    SignalAgent
        ↓
Image ECG → Segmentation nnU-Net/M3
          → Digitalisation
          → Signal numérique
          → ResNet1D
        ↓
LiteratureAgent + WebSearchAgent
        ↓
   ClinicalAgent
        ↓
    CriticAgent
        ↓
 SynthesizerAgent
        ↓
 Final Critic
        ↓
 ConfidenceAgent
        ↓
 Rapport clinique final
```

Les classes ECG utilisées sont :

```text
NORM | MI | STTC | CD | HYP
```

La sortie du modèle ECG est considérée comme une **prédiction**, jamais comme un diagnostic clinique confirmé.

## Structure principale

```text
CardioVision-IA/
│
├── agents/
├── orchestrator/
├── rag/
├── rag_ecg_reference/
├── rag_ptbxl/
├── rag_research/
├── evaluation/
├── tests/
│
├── ecg_project/
│   ├── ecg_agent.py
│   ├── tools/
│   ├── src/
│   ├── utils/
│   ├── nnUNet/
│   └── models/
│
├── app_streamlit_native.py
├── pipeline_worker.py
├── requirements.txt
└── README.md
```

## Modèles à télécharger

Les poids volumineux ne sont pas inclus dans le dépôt GitHub.

Téléchargement :

**[Google Drive — Modèles CardioVision IA](https://drive.google.com/drive/folders/1btz7o3BiCI_gIA6z7pvfxHB-V8AzhOto?usp=sharing)**

Trois fichiers doivent être ajoutés.

### ResNet1D

À placer dans :

```text
ecg_project/models/
```

Fichiers :

```text
best_resnet1d_wang_standardscaler.pt
best_resnet1d_wang_finetuned_reconstructed.pt
```

Le fichier :

```text
global_standard_scaler.json
```

est déjà présent dans le dépôt.

### M3 — segmentation ECG

Le modèle :

```text
checkpoint_best.pth
```

doit être placé dans :

```text
ecg_project/models/M3/nnUNet_results/
└── Dataset500_Signals/
    └── nnUNetTrainer__nnUNetPlans__2d/
        └── fold_all/
            └── checkpoint_best.pth
```

Les fichiers `dataset.json`, `dataset_fingerprint.json` et `plans.json` sont déjà présents dans le dépôt.

## Installation

```bash
git clone https://github.com/ahmed10420/CardioVision-IA.git
cd CardioVision-IA
```

Créer un environnement virtuel :

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Installer les dépendances :

```powershell
pip install -r requirements.txt
```

Installer ensuite les dépendances ECG et le fork local de nnU-Net :

```powershell
cd ecg_project
pip install -r requirements.txt
pip install -e .\nnUNet
cd ..
```

Configurer ensuite les clés API dans un fichier `.env`.

## Lancer l’interface

Depuis la racine du projet :

```powershell
python -m streamlit run app_streamlit_native.py
```

L’interface permet de saisir un cas clinique, ajouter éventuellement une image ECG, lancer le pipeline multi-agent et consulter le rapport final ainsi que les résultats du SignalAgent et du RAG.

## Sources médicales

Le système utilise plusieurs types de sources :

- guidelines ESC/AHA ;
- références ECG ;
- littérature scientifique ;
- cas PTB-XL ;
- recherche Web complémentaire.

Les cas PTB-XL sont utilisés uniquement comme références comparatives et ne confirment pas un diagnostic.

## ECG-Digitiser

La partie de digitalisation ECG réutilise et adapte plusieurs composants du projet open-source **ECG-Digitiser**, développé pour le George B. Moody PhysioNet Challenge 2024.

Projet original :

https://github.com/felixkrones/ECG-Digitiser

Référence :

Felix Krones, Ben Walker, Terry Lyons, Adam Mahdi.  
**Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts.**  
arXiv:2410.14185, 2024.

https://arxiv.org/abs/2410.14185

## Technologies

Python, PyTorch, ResNet1D, nnU-Net, LangGraph, LangChain, Streamlit, Chroma, Sentence Transformers, PTB-XL, WFDB, Groq, NVIDIA NIM et Gemini.

## Projet

**CardioVision IA — Multi-Agent ECG Clinical Decision Support**

Projet réalisé dans le cadre d’un stage de recherche au :

**CES-Lab — École Nationale d’Ingénieurs de Sfax (ENIS)**

**Auteur : Ahmed Gaabi**  
**Encadré par : M. Ayman Abid**
