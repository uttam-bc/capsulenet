"""
Response formatter for the Offline Internet Capsule.
Generates structured, template-based answers from search results.
"""

from core.search import SearchResult


def format_response(query: str, results: list[SearchResult]) -> dict:
    """
    Format search results into a structured, human-readable answer.

    Returns:
        dict with keys: answer (str), sources (list), confidence (float), category (str|None)
    """
    if not results:
        return {
            "answer": _no_results_answer(query),
            "sources": [],
            "confidence": 0.0,
            "category": None,
        }

    top = results[0]
    confidence = _calculate_confidence(results)
    category = top.category

    # Build answer based on category
    if category == "medical":
        answer = _format_medical(query, results)
    elif category == "survival":
        answer = _format_survival(query, results)
    elif category == "navigation":
        answer = _format_navigation(query, results)
    elif category == "education":
        answer = _format_education(query, results)
    else:
        answer = _format_general(query, results)

    # Build sources list
    sources = [
        {
            "title": r.title,
            "category": r.category,
            "relevance": round(min(r.score, 1.0), 2),
        }
        for r in results[:3]
    ]

    return {
        "answer": answer,
        "sources": sources,
        "confidence": round(confidence, 2),
        "category": category,
    }


def _calculate_confidence(results: list[SearchResult]) -> float:
    """Calculate overall confidence based on top result scores."""
    if not results:
        return 0.0
    top_score = results[0].score
    # Scale: high scores → high confidence, with diminishing returns
    confidence = min(top_score * 1.2, 1.0)
    # Boost if multiple results agree on topic
    if len(results) >= 2 and results[1].score > 0.3:
        confidence = min(confidence + 0.05, 1.0)
    return confidence


def _format_medical(query: str, results: list[SearchResult]) -> str:
    """Format a medical response with warnings and action steps."""
    top = results[0]
    content = top.content

    lines = [f"## {top.title}\n"]

    # Add warning header for medical content
    lines.append("⚠️ **IMPORTANT**: This information is for educational purposes only. "
                 "Always seek professional medical help when possible.\n")

    # Split content into structured sections
    lines.append(_structure_content(content))

    # Add related topics
    if len(results) > 1:
        lines.append("\n**Related Topics:**")
        for r in results[1:3]:
            lines.append(f"- {r.title}")

    return "\n".join(lines)


def _format_survival(query: str, results: list[SearchResult]) -> str:
    """Format a survival response with step-by-step instructions."""
    top = results[0]
    content = top.content

    lines = [f"## 🏕️ {top.title}\n"]

    lines.append(_structure_content(content))

    if len(results) > 1:
        lines.append("\n**See Also:**")
        for r in results[1:3]:
            lines.append(f"- {r.title}")

    return "\n".join(lines)


def _format_navigation(query: str, results: list[SearchResult]) -> str:
    """Format a navigation response with clear directional guidance."""
    top = results[0]
    content = top.content

    lines = [f"## 🧭 {top.title}\n"]

    lines.append(_structure_content(content))

    if len(results) > 1:
        lines.append("\n**Related Navigation Guides:**")
        for r in results[1:3]:
            lines.append(f"- {r.title}")

    return "\n".join(lines)


def _format_education(query: str, results: list[SearchResult]) -> str:
    """Format an educational response with clear explanations."""
    top = results[0]
    content = top.content

    lines = [f"## 📚 {top.title}\n"]

    lines.append(_structure_content(content))

    if len(results) > 1:
        lines.append("\n**Learn More:**")
        for r in results[1:3]:
            lines.append(f"- {r.title}")

    return "\n".join(lines)


def _format_general(query: str, results: list[SearchResult]) -> str:
    """Format a general response."""
    top = results[0]
    lines = [f"## {top.title}\n"]
    lines.append(_structure_content(top.content))
    return "\n".join(lines)


def _structure_content(content: str) -> str:
    """Convert raw content into structured bullet points and sections."""
    # Split by numbered steps (1), 2), etc.) or by sentences with markers
    structured_lines = []

    # Handle WARNING sections
    parts = content.split("WARNING:")
    main_content = parts[0]
    warning = parts[1].strip() if len(parts) > 1 else None

    # Handle CRITICAL sections
    critical = None
    critical_parts = main_content.split("CRITICAL:")
    if len(critical_parts) > 1:
        main_content = critical_parts[0]
        critical = critical_parts[1].strip()

    # Split main content into sentences/steps
    sentences = _smart_split(main_content)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Detect numbered steps
        if sentence[0].isdigit() and ")" in sentence[:4]:
            structured_lines.append(f"  {sentence}")
        elif sentence.startswith(("a)", "b)", "c)", "d)", "e)")):
            structured_lines.append(f"    {sentence}")
        else:
            structured_lines.append(f"- {sentence}")

    result = "\n".join(structured_lines)

    if critical:
        result += f"\n\n🚨 **CRITICAL**: {critical}"

    if warning:
        result += f"\n\n⚠️ **WARNING**: {warning}"

    return result


def _smart_split(text: str) -> list[str]:
    """Split content intelligently at sentence boundaries or numbered steps."""
    import re

    # Split at numbered step boundaries: "1)", "2)", etc.
    parts = re.split(r'(?=\d+\))', text)

    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # If it's a numbered step, keep it whole
        if part[0].isdigit() and ")" in part[:4]:
            # But split sub-steps (a), b), etc.)
            sub_parts = re.split(r'(?=[a-e]\))', part)
            result.extend(sub_parts)
        else:
            # Split long non-step text into sentences
            sentences = re.split(r'(?<=[.!])\s+', part)
            result.extend(sentences)

    return result


def _no_results_answer(query: str) -> str:
    """Generate a helpful response when no results are found."""
    return (
        f"## No Results Found\n\n"
        f"I couldn't find relevant information for **\"{query}\"**.\n\n"
        f"**Suggestions:**\n"
        f"- Try rephrasing your question with different keywords\n"
        f"- Browse available categories using the category pills above\n"
        f"- Check the emergency quick-access section for critical information\n\n"
        f"*This offline knowledge base covers: Medical, Survival, Navigation, and Education topics.*"
    )


def format_emergency_cards() -> list[dict]:
    """Generate pre-built emergency quick-access cards."""
    return [
        {
            "id": "emg-cpr",
            "title": "CPR Guide",
            "icon": "❤️",
            "category": "medical",
            "steps": [
                "Check responsiveness - tap and shout",
                "Call for help",
                "Place on firm surface, on their back",
                "30 chest compressions (push hard, 2 inches deep)",
                "2 rescue breaths (tilt head, lift chin)",
                "Repeat 30:2 cycle until help arrives",
            ],
            "warning": "Do NOT perform on someone breathing normally",
        },
        {
            "id": "emg-burns",
            "title": "Burn Treatment",
            "icon": "🔥",
            "category": "medical",
            "steps": [
                "Cool under running water for 10-20 minutes",
                "Do NOT apply ice, butter, or toothpaste",
                "Remove jewelry before swelling",
                "Apply aloe vera gel",
                "Cover loosely with sterile bandage",
                "For severe burns: seek emergency care",
            ],
            "warning": "Never pop blisters — infection risk",
        },
        {
            "id": "emg-snake",
            "title": "Snake Bite",
            "icon": "🐍",
            "category": "medical",
            "steps": [
                "Move away from the snake",
                "Keep victim calm and still",
                "Remove jewelry near bite",
                "Position bitten limb below heart level",
                "Apply pressure bandage from bite toward heart",
                "Transport to medical facility ASAP",
            ],
            "warning": "Do NOT cut wound, suck venom, apply tourniquet, or ice",
        },
        {
            "id": "emg-choking",
            "title": "Choking Response",
            "icon": "😮",
            "category": "medical",
            "steps": [
                "Ask 'Are you choking?' — act if they can't speak",
                "Stand behind, wrap arms around waist",
                "Fist above navel, below ribcage",
                "Quick upward thrusts",
                "Repeat until object expelled",
                "If unconscious: begin CPR",
            ],
            "warning": "For infants: 5 back blows + 5 chest thrusts instead",
        },
        {
            "id": "emg-bleeding",
            "title": "Severe Bleeding",
            "icon": "🩸",
            "category": "medical",
            "steps": [
                "Apply firm direct pressure with clean cloth",
                "If blood soaks through, add layers — don't remove first",
                "Elevate wounded area above heart",
                "For life-threatening limb bleeding: apply tourniquet",
                "Note time of tourniquet application",
                "Keep victim warm, watch for shock",
            ],
            "warning": "Deep puncture wounds need professional care",
        },
        {
            "id": "emg-water",
            "title": "Water Purification",
            "icon": "💧",
            "category": "survival",
            "steps": [
                "BOILING: Rolling boil for 1+ minute",
                "SOLAR: Clear bottle in sun for 6+ hours",
                "CHEMICAL: Purification tablets, 30 min wait",
                "FILTER: Gravel → sand → charcoal layers",
                "Always filter cloudy water before purifying",
                "Never drink saltwater or chemically contaminated water",
            ],
            "warning": "Unpurified water can cause deadly illness",
        },
        {
            "id": "emg-earthquake",
            "title": "Earthquake Safety",
            "icon": "🏚️",
            "category": "survival",
            "steps": [
                "DROP to hands and knees",
                "Take COVER under sturdy furniture",
                "HOLD ON and protect head/neck",
                "Stay inside until shaking stops",
                "If outdoors: move to open area",
                "If near coast: move to high ground (tsunami risk)",
            ],
            "warning": "Expect aftershocks — stay alert",
        },
        {
            "id": "emg-hypothermia",
            "title": "Hypothermia Response",
            "icon": "🥶",
            "category": "survival",
            "steps": [
                "Get person out of cold/wet environment",
                "Remove wet clothing, replace with dry",
                "Warm center of body first (chest, neck, groin)",
                "Use skin-to-skin contact in blankets",
                "Give warm drinks if conscious (no alcohol)",
                "Handle gently — avoid rough movement",
            ],
            "warning": "Severe hypothermia needs emergency medical care",
        },
    ]
