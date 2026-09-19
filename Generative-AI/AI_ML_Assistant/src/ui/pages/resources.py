"""📚 Stacks workspace: curated crash courses and the RAG/tooling docs."""

from __future__ import annotations

import html

import streamlit as st

from src.ui.registry import register_page

# Curated learning resources: subject -> list of (title, url, blurb).
CRASH_COURSES: dict[str, list[tuple[str, str, str]]] = {
    "Machine Learning": [
        (
            "Machine Learning Specialization — Andrew Ng",
            "https://www.coursera.org/specializations/machine-learning-introduction",
            "The classic ground-up course: regression, classification, trees, best practices.",
        ),
        (
            "Google ML Crash Course",
            "https://developers.google.com/machine-learning/crash-course",
            "Fast, free, and hands-on with short videos and exercises.",
        ),
        (
            "StatQuest (Josh Starmer)",
            "https://www.youtube.com/@statquest",
            "The friendliest intuition-first explanations of ML and statistics on YouTube.",
        ),
        (
            "scikit-learn User Guide",
            "https://scikit-learn.org/stable/user_guide.html",
            "Practical reference for every classical model with worked examples.",
        ),
    ],
    "Deep Learning": [
        (
            "Stanford CS231n — Deep Learning for Computer Vision",
            "https://cs231n.stanford.edu",
            "Stanford's flagship DL course: CNNs, backprop, and training, with public "
            "lecture notes and assignments — the standard academic reference.",
        ),
        (
            "Deep Learning Specialization — DeepLearning.AI",
            "https://www.coursera.org/specializations/deep-learning",
            "Neural networks, CNNs, sequence models, and tuning — the standard DL path.",
        ),
        (
            "Neural Networks: Zero to Hero — Andrej Karpathy",
            "https://karpathy.ai/zero-to-hero.html",
            "Build backprop, then GPT, from scratch in code. Best hands-on DL series.",
        ),
        (
            "Dive into Deep Learning (d2l.ai)",
            "https://d2l.ai",
            "Free interactive textbook: maths, code, and exercises for every architecture.",
        ),
        (
            "3Blue1Brown — Neural networks",
            "https://www.3blue1brown.com/topics/neural-networks",
            "Beautiful visual intuition for what networks and gradient descent really do.",
        ),
        (
            "PyTorch & torchvision — official tutorials",
            "https://pytorch.org/tutorials/",
            "The research-favourite framework, hands-on; torchvision adds vision datasets, "
            "pretrained models, and image transforms.",
        ),
        (
            "TensorFlow & Keras — tutorials",
            "https://www.tensorflow.org/tutorials",
            "Google's production DL stack; Keras is the high-level API for building and "
            "training models quickly.",
        ),
    ],
    "NLP": [
        (
            "Stanford CS224N — NLP with Deep Learning",
            "https://web.stanford.edu/class/cs224n/",
            "The standard academic NLP course: word vectors, attention, transformers, and "
            "pretraining, with public slides and lectures.",
        ),
        (
            "Hugging Face NLP Course",
            "https://huggingface.co/learn/nlp-course",
            "Transformers, tokenizers, and fine-tuning end-to-end in code with the HF stack.",
        ),
        (
            "Speech and Language Processing — Jurafsky & Martin",
            "https://web.stanford.edu/~jurafsky/slp3/",
            "The free, comprehensive NLP textbook, from n-grams to transformers and dialogue.",
        ),
        (
            "Advanced NLP with spaCy",
            "https://course.spacy.io",
            "Free hands-on course for building real, production NLP pipelines with spaCy.",
        ),
    ],
    "AI Engineering & LLMs": [
        (
            "Full Stack LLM Bootcamp",
            "https://fullstackdeeplearning.com/llm-bootcamp/",
            "Free, practical course on building and shipping LLM apps: prompting, retrieval, "
            "UX, and deployment.",
        ),
        (
            "DeepLearning.AI Short Courses",
            "https://www.deeplearning.ai/short-courses/",
            "1-hour courses on RAG, LangChain, prompt engineering, evals, and agents.",
        ),
        (
            "Prompt Engineering Guide",
            "https://www.promptingguide.ai",
            "Techniques (CoT, few-shot, self-consistency) with papers and examples.",
        ),
        (
            "fast.ai — Practical Deep Learning",
            "https://course.fast.ai",
            "Top-down: train state-of-the-art models first, learn the theory as you go.",
        ),
    ],
    "Quantum Machine Learning": [
        (
            "Quantum Machine Learning — Biamonte et al. (2017)",
            "https://arxiv.org/abs/1611.09347",
            "The standard Nature review mapping out the field: where quantum computing and "
            "machine learning meet, and what speed-ups are (and aren't) plausible.",
        ),
        (
            "PennyLane — Quantum ML (Xanadu)",
            "https://pennylane.ai/qml/",
            "The leading QML platform: differentiable quantum circuits with a large library "
            "of tutorials and demos bridging machine learning and quantum computing.",
        ),
        (
            "Qiskit Machine Learning (IBM)",
            "https://qiskit-community.github.io/qiskit-machine-learning/",
            "IBM's library for quantum kernels, variational classifiers, and QNNs on "
            "simulators and real hardware.",
        ),
        (
            "TensorFlow Quantum",
            "https://www.tensorflow.org/quantum",
            "Google's library for hybrid quantum-classical models, blending Cirq circuits "
            "with Keras layers.",
        ),
    ],
    "Neural Quantum States": [
        (
            "Neural-network quantum states — Carleo & Troyer (2017)",
            "https://arxiv.org/abs/1606.02318",
            "The Science paper that launched the field: solving the quantum many-body "
            "problem for condensed-matter systems with artificial neural networks.",
        ),
        (
            "NetKet — neural quantum states & VMC",
            "https://www.netket.org",
            "The standard open-source framework for machine-learning quantum many-body "
            "systems: neural quantum states and variational Monte Carlo on a lattice.",
        ),
        (
            "NetKet — fermions & variational Monte Carlo (docs)",
            "https://netket.readthedocs.io",
            "Practical guide to building second-quantized fermionic Hamiltonians (incl. the "
            "Hubbard model) and running VMC with neural quantum states.",
        ),
        (
            "jVMC — GPU variational Monte Carlo in JAX",
            "https://github.com/markusschmitt/vmc_jax",
            "GPU-accelerated VMC with neural quantum states for lattice spin and fermion "
            "models — a fast, research-grade complement to NetKet.",
        ),
        (
            "Fermionic neural-network states — Choo, Mezzacapo & Carleo (2020)",
            "https://arxiv.org/abs/1909.12852",
            "Maps fermions onto neural quantum states via Jordan–Wigner, enabling VMC for "
            "fermionic Hamiltonians — the reference method for lattice fermionic NQS.",
        ),
        (
            "Neural backflow for fermions — Luo & Clark (2019)",
            "https://arxiv.org/abs/1807.10770",
            "Neural-network backflow transformations for fermionic wavefunctions, "
            "benchmarked in VMC on the Hubbard model.",
        ),
        (
            "Hidden-fermion determinant states — Moreno et al. (2022)",
            "https://arxiv.org/abs/2111.10420",
            "Neural-network fermionic wavefunctions with hidden fermions, achieving "
            "state-of-the-art VMC energies on the 2D Hubbard model.",
        ),
        (
            "FermiNet — Pfau et al. (2020)",
            "https://arxiv.org/abs/1909.02487",
            "DeepMind's deep-neural-network ansatz for continuum (ab-initio) fermionic VMC, "
            "solving the many-electron Schrödinger equation directly in real space.",
        ),
        (
            "PauliNet — Hermann, Schätzle & Noé (2020)",
            "https://arxiv.org/abs/1909.08423",
            "Deep-learning real-space fermionic wavefunction that builds in physical "
            "constraints (cusps, antisymmetry) for efficient molecular VMC.",
        ),
        (
            "DeepQMC — deep-learning VMC toolkit",
            "https://github.com/deepqmc/deepqmc",
            "PyTorch library implementing PauliNet/FermiNet-style ansätze for real-space "
            "variational and diffusion Monte Carlo.",
        ),
    ],
}

# "Go deeper" per subject: the tools, references, and foundational papers that pair with
# each crash course, ordered **beginner → advanced** so a reader knows where to step next.
# Keys match CRASH_COURSES exactly. Even-length lists keep the two-column grid tidy.
# The AI Engineering & LLMs entry doubles as the docs behind *this* app (formerly a single
# shared footer) — folded in here because that is the only subject it is genuinely relevant to.
GO_DEEPER: dict[str, list[tuple[str, str, str]]] = {
    "Machine Learning": [
        (
            "Kaggle Learn — Intro to Machine Learning",
            "https://www.kaggle.com/learn/intro-to-machine-learning",
            "Free micro-course: train your first models hands-on in a browser notebook.",
        ),
        (
            "pandas — documentation",
            "https://pandas.pydata.org/docs/",
            "The data-wrangling library you reach for before any model — official user guide.",
        ),
        (
            "An Introduction to Statistical Learning (ISLR)",
            "https://www.statlearning.com",
            "The friendly, free textbook on the theory behind classical ML, with code labs.",
        ),
        (
            "The Elements of Statistical Learning (ESL)",
            "https://hastie.su.domains/ElemStatLearn/",
            "The advanced reference: the maths behind the methods, free from the authors.",
        ),
    ],
    "Deep Learning": [
        (
            "TensorFlow Playground",
            "https://playground.tensorflow.org",
            "Train a tiny neural net in your browser and watch it learn — pure intuition.",
        ),
        (
            "Weights & Biases — documentation",
            "https://docs.wandb.ai",
            "Track experiments, hyperparameters, and metrics — the standard ML-ops tool.",
        ),
        (
            "Deep Learning — Goodfellow, Bengio & Courville",
            "https://www.deeplearningbook.org",
            "The comprehensive, free textbook on the theory underpinning deep learning.",
        ),
        (
            "Papers with Code",
            "https://paperswithcode.com",
            "State-of-the-art results linked to runnable code, browsable by task and benchmark.",
        ),
    ],
    "NLP": [
        (
            "The Illustrated Transformer — Jay Alammar",
            "https://jalammar.github.io/illustrated-transformer/",
            "The clearest visual walk-through of attention and the transformer block.",
        ),
        (
            "NLTK Book",
            "https://www.nltk.org/book/",
            "Classic hands-on intro to text processing and linguistics in Python.",
        ),
        (
            "BERT — Devlin et al., 2018",
            "https://arxiv.org/abs/1810.04805",
            "The pretraining paper that reshaped modern NLP.",
        ),
        (
            "GPT-3 — Brown et al., 2020",
            "https://arxiv.org/abs/2005.14165",
            "'Language Models are Few-Shot Learners' — the scaling paper behind modern LLMs.",
        ),
    ],
    "AI Engineering & LLMs": [
        (
            "LangChain documentation",
            "https://python.langchain.com",
            "The orchestration framework this app uses for LLM calls, tools, and retrievers.",
        ),
        (
            "Chroma documentation",
            "https://docs.trychroma.com",
            "The local vector store holding this app's knowledge-base embeddings.",
        ),
        (
            "Sentence-Transformers (SBERT)",
            "https://www.sbert.net",
            "The local embedding models used for free, offline ingestion.",
        ),
        (
            "OpenRouter documentation",
            "https://openrouter.ai/docs",
            "The multi-model API gateway this app calls for LLM completions.",
        ),
        (
            "RAG paper — Lewis et al., 2020",
            "https://arxiv.org/abs/2005.11401",
            "'Retrieval-Augmented Generation…' — the pattern this very app is built on.",
        ),
        (
            "Attention Is All You Need — Vaswani et al., 2017",
            "https://arxiv.org/abs/1706.03762",
            "The transformer paper behind every modern LLM.",
        ),
    ],
    "Quantum Machine Learning": [
        (
            "Quantum Country — Nielsen & Matuschak",
            "https://quantum.country",
            "A beautiful, memorable introduction to qubits and quantum algorithms.",
        ),
        (
            "Cirq — documentation",
            "https://quantumai.google/cirq",
            "Google's framework for writing and simulating quantum circuits in Python.",
        ),
        (
            "Variational Quantum Algorithms — Cerezo et al., 2021",
            "https://arxiv.org/abs/2012.09265",
            "The review of the near-term variational algorithms that power much of QML.",
        ),
        (
            "Quantum Computation and Quantum Information — Nielsen & Chuang",
            "https://en.wikipedia.org/wiki/Quantum_Computation_and_Quantum_Information",
            "The standard graduate textbook underpinning the whole field.",
        ),
    ],
    "Neural Quantum States": [
        (
            "JAX — documentation",
            "https://jax.readthedocs.io",
            "Autodiff + XLA on GPU/TPU — the numerical backbone of modern NQS codes.",
        ),
        (
            "Flax — neural network library",
            "https://flax.readthedocs.io",
            "The JAX network library NetKet uses to define neural quantum-state ansätze.",
        ),
        (
            "QuSpin — exact diagonalization",
            "https://quspin.github.io/QuSpin/",
            "Exact many-body reference results to benchmark your variational NQS against.",
        ),
        (
            "Quantum Monte Carlo Approaches — Becca & Sorella (2017)",
            "https://www.cambridge.org/core/books/quantum-monte-carlo-approaches-for-correlated-systems/EB88C86BD9553A0738BDAE400D0B2900",
            "The reference text on variational Monte Carlo for correlated lattice systems.",
        ),
    ],
}


def _resource_card_html(title: str, url: str, blurb: str) -> str:
    """Build one resource card's HTML, escaping every interpolated value.

    All current callers pass hardcoded constants, but escaping here is a standing guardrail:
    if a live/model/retrieval-derived value is ever fed in, it can't inject markup or script.
    """
    return (
        f'<div class="resource-card"><h4>'
        f'<a href="{html.escape(url, quote=True)}" target="_blank">{html.escape(title)}</a>'
        f"</h4><p>{html.escape(blurb)}</p></div>"
    )


def _resource_card(title: str, url: str, blurb: str) -> None:
    """Render one styled, clickable resource card."""
    st.markdown(_resource_card_html(title, url, blurb), unsafe_allow_html=True)


def _card_grid(sources: list[tuple[str, str, str]]) -> None:
    """Lay out a list of ``(title, url, blurb)`` resources as a two-column card grid."""
    col_left, col_right = st.columns(2)
    for i, (title, url, blurb) in enumerate(sources):
        with col_left if i % 2 == 0 else col_right:
            _resource_card(title, url, blurb)


@register_page("📚 Stacks", key="research", section="Knowledge", order=75)
def render() -> None:
    """Stacks tab: crash courses plus subject-specific tools and deeper reading."""
    st.subheader("📚 Stacks")
    st.caption(
        "Curated **courses, tools & references** per subject — evergreen picks to learn from "
        "and go deeper. For guided in-app lessons see **🎓 AI/ML Tutor**; for live papers, "
        "**📰 AI News**."
    )

    st.markdown("### 🎓 Crash courses")
    st.caption("Hand-picked starting points, from absolute beginner to hands-on builder.")
    tabs = st.tabs(list(CRASH_COURSES))
    for tab, (subject, courses) in zip(tabs, CRASH_COURSES.items(), strict=True):
        with tab:
            for title, url, blurb in courses:
                _resource_card(title, url, blurb)

            deeper = GO_DEEPER.get(subject)
            if deeper:
                st.markdown("#### 🧰 Tools & deeper reading")
                st.caption("Where to go next for this subject — beginner → advanced.")
                _card_grid(deeper)
