"""
formatter.py
Post-processes Gemma 4 output for display. Handles language fallbacks.
"""


def format_results(results: list[dict], lang_code: str) -> list[dict]:
    """Ensure all result fields are display-ready strings."""
    for r in results:
        if not r.get("match_label"):
            r["match_label"] = "POSSIBLE"
        if not r.get("match_reason"):
            r["match_reason"] = "Eligibility could not be determined automatically."
        if not r.get("plain_summary"):
            r["plain_summary"] = "No summary available."
        if not isinstance(r.get("key_criteria"), list):
            r["key_criteria"] = []
        if not r.get("doctor_questions"):
            r["doctor_questions"] = _default_questions(lang_code)
    return results


def _default_questions(lang_code: str) -> list[str]:
    questions = {
        "en": [
            "Do you think I might qualify for any clinical trials given my diagnosis?",
            "What are the main risks and benefits of participating in a trial?",
            "How would participating affect my current treatment plan?",
            "Will I need to travel frequently, and is transportation support available?",
            "What happens to my care if I withdraw from the trial?",
        ],
        "pt": [
            "Você acha que posso me qualificar para algum ensaio clínico dado meu diagnóstico?",
            "Quais são os principais riscos e benefícios de participar de um ensaio?",
            "Como a participação afetaria meu plano de tratamento atual?",
            "Precisarei viajar com frequência e há suporte de transporte disponível?",
            "O que acontece com meu atendimento se eu sair do ensaio?",
        ],
        "es": [
            "¿Cree que podría calificar para algún ensayo clínico dado mi diagnóstico?",
            "¿Cuáles son los principales riesgos y beneficios de participar en un ensayo?",
            "¿Cómo afectaría la participación mi plan de tratamiento actual?",
            "¿Necesitaré viajar con frecuencia y hay apoyo de transporte disponible?",
            "¿Qué pasa con mi atención si me retiro del ensayo?",
        ],
        "fr": [
            "Pensez-vous que je pourrais être éligible à un essai clinique?",
            "Quels sont les principaux risques et avantages de participer à un essai?",
            "Comment la participation affecterait-elle mon plan de traitement actuel?",
            "Devrai-je voyager fréquemment et un soutien au transport est-il disponible?",
            "Que se passe-t-il pour mes soins si je me retire de l'essai?",
        ],
    }
    return questions.get(lang_code, questions["en"])
