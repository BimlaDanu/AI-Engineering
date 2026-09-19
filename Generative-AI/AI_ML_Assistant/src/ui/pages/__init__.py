"""Importing this package registers every workspace page with the registry.

Add a new workspace by creating a module here that decorates its ``render`` function
with :func:`src.ui.registry.register_page`, then import it below.
"""

from src.ui.pages import (  # noqa: F401  (imported for their registration side effects)
    ab_testing,  # registers the 🆚 A/B testing page (two RAG strategies head-to-head)
    ai_news,
    analytics,
    chat,
    evaluation,  # registers the 📊 Evaluation page (RAGAs-style metrics)
    home,
    inspector,  # registers the 🧪 Experiments page (also pulls in the tool playground)
    knowledge_base,
    ml_lab,  # registers the 🔬 AI/ML Lab page (hands-on ML/DL/NLP recipes + code cell)
    ml_tutor,
    quiz,  # registers the 🎯 Trivia page (KB-grounded MCQ quiz)
    resources,  # registers the 📚 Stacks page
    settings,
)
