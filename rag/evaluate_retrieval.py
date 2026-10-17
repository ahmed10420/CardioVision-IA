from rag.retriever import retrieve_documents


TEST_CASES = [
    {
        "question": "atrial fibrillation diagnosis electrocardiogram ECG confirmation irregular RR intervals absence of P waves ESC 2024",
        "expected_source": "esc_afib_guidelines_2024.pdf",
        "positive_keywords": [
            "atrial fibrillation",
            "diagnosis",
            "ecg",
            "confirmation",
            "irregular",
            "p waves",
        ],
        "negative_keywords": [
            "doi.org",
            "references",
            "et al.",
            "downloaded from",
        ],
    },
    {
        "question": "STEMI ECG diagnostic criteria ST elevation contiguous leads ESC 2023",
        "expected_source": "esc_acute_coronary_syndromes_2023.pdf",
        "positive_keywords": [
            "st-segment elevation",
            "contiguous leads",
            "stemi",
            "ecg",
            "diagnosis",
        ],
        "negative_keywords": [
            "doi.org",
            "references",
            "et al.",
            "downloaded from",
        ],
    },
    {
        "question": "chronic coronary syndrome ECG initial evaluation 12-lead ECG ESC 2024",
        "expected_source": "esc_chronic_coronary_syndromes_2024.pdf",
        "positive_keywords": [
            "12-lead ecg",
            "initial evaluation",
            "chronic coronary syndrome",
            "chest pain",
        ],
        "negative_keywords": [
            "doi.org",
            "references",
            "et al.",
            "downloaded from",
        ],
    },
]


def keyword_score(text, positive_keywords, negative_keywords):
    text_lower = text.lower()

    positive_hits = sum(1 for kw in positive_keywords if kw.lower() in text_lower)
    negative_hits = sum(1 for kw in negative_keywords if kw.lower() in text_lower)

    score = positive_hits - negative_hits

    return score, positive_hits, negative_hits


def evaluate_case(case, k=5):
    question = case["question"]
    results = retrieve_documents(question, k=k)

    expected_source = case["expected_source"]

    hit_at_k = 0
    reciprocal_rank = 0
    useful_count = 0

    print("\n" + "=" * 80)
    print("QUESTION :", question)
    print("SOURCE ATTENDUE :", expected_source)
    print("=" * 80)

    for rank, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "")
        text = doc.page_content

        score, pos_hits, neg_hits = keyword_score(
            text,
            case["positive_keywords"],
            case["negative_keywords"],
        )

        is_expected_source = expected_source in source
        is_useful = score > 0 and is_expected_source

        if is_expected_source and hit_at_k == 0:
            hit_at_k = 1
            reciprocal_rank = 1 / rank

        if is_useful:
            useful_count += 1

        print(f"\nRang {rank}")
        print("Source :", source)
        print("Page :", doc.metadata.get("page"))
        print("Score mots-clés :", score)
        print("Positive hits :", pos_hits)
        print("Negative hits :", neg_hits)
        print("Utile :", is_useful)
        print("Extrait :", text[:300].replace("\n", " "))

    precision_at_k = useful_count / k

    return {
        "question": question,
        "hit_at_k": hit_at_k,
        "mrr": reciprocal_rank,
        "precision_at_k": precision_at_k,
        "useful_count": useful_count,
        "k": k,
    }


def main():
    k = 5
    all_metrics = []

    for case in TEST_CASES:
        metrics = evaluate_case(case, k=k)
        all_metrics.append(metrics)

    print("\n" + "=" * 80)
    print("RÉSULTATS GLOBAUX")
    print("=" * 80)

    avg_hit = sum(m["hit_at_k"] for m in all_metrics) / len(all_metrics)
    avg_mrr = sum(m["mrr"] for m in all_metrics) / len(all_metrics)
    avg_precision = sum(m["precision_at_k"] for m in all_metrics) / len(all_metrics)

    print(f"Hit@{k} moyen :", round(avg_hit, 3))
    print("MRR moyen :", round(avg_mrr, 3))
    print(f"Precision@{k} moyenne :", round(avg_precision, 3))


if __name__ == "__main__":
    main()