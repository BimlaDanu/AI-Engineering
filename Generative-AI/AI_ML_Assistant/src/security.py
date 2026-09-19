"""Input validation, prompt-injection screening, and ML/AI domain gating.

Layered defence (OWASP LLM01): these patterns, then the LLM classifier in
:meth:`src.core.router.QueryRouter.screen_injection`, then system-prompt hardening.

Limitations: the classifier fails open; only user input is screened, not retrieved or tool
content; patterns are English and literal, so translation or encoding evades them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_INPUT_CHARS = 2000

# Patterns that commonly appear in prompt-injection attempts. Pattern matching is a first
# line of defence; the system prompt independently instructs the model to ignore embedded
# instructions (defence in depth).
_INJECTION_PATTERNS = [
    r"ignore (all |any |your )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |your )?(previous|prior|system) (instructions|prompts?)",
    r"you are now\b",
    r"pretend (to be|you are)",
    r"reveal (your|the) (instructions|prompt|rules)",
    r"</?(system|assistant|instructions?)>",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Vocabulary signalling the question is inside the ML/AI domain. Used as a fallback when
# retrieval scores are low (e.g. pure tool requests like "estimate tokens for this prompt").
_DOMAIN_HINTS = {
    "ml",
    "ai",
    "machine",
    "learning",
    "deep",
    "model",
    "models",
    "neural",
    "network",
    "networks",
    "transformer",
    "transformers",
    "attention",
    "embedding",
    "embeddings",
    "vector",
    "vectors",
    "gradient",
    "descent",
    "regression",
    "classification",
    "clustering",
    "overfitting",
    "underfitting",
    "regularization",
    "regularisation",
    "rag",
    "retrieval",
    "llm",
    "llms",
    "prompt",
    "prompts",
    "prompting",
    "fine-tuning",
    "finetuning",
    "lora",
    "rlhf",
    "token",
    "tokens",
    "tokenizer",
    "loss",
    "optimizer",
    "optimization",
    "backpropagation",
    "dataset",
    "datasets",
    "training",
    "inference",
    "precision",
    "recall",
    "accuracy",
    "f1",
    "cross-validation",
    "bayes",
    "bayesian",
    "probability",
    "statistics",
    "matrix",
    "eigenvalue",
    "eigenvalues",
    "cosine",
    "similarity",
    "arxiv",
    "paper",
    "papers",
    "bert",
    "gpt",
    "diffusion",
    "reinforcement",
    "agent",
    "agents",
    "langchain",
    "chroma",
    "huggingface",
    "hyperparameter",
    "hyperparameters",
    "feature",
    "features",
    "bias",
    "variance",
    "svm",
    "kernel",
    "tree",
    "forest",
    "boosting",
    "cnn",
    "rnn",
    "lstm",
    "gan",
    "autoencoder",
    "perceptron",
    "softmax",
    "sigmoid",
    "relu",
    "batch",
    "epoch",
    "weights",
    "convolution",
    "convolutional",
    "pooling",
    "dropout",
    "batchnorm",
    "normalization",
    "activation",
    "activations",
    "backprop",
    "vanishing",
    "exploding",
    "residual",
    "resnet",
    "parameters",
    "quantization",
    "distillation",
    "hallucination",
    "benchmark",
    "evaluation",
    "api",
    "apis",
    "sdk",
    "sdks",
    "mcp",
    "endpoint",
    "endpoints",
    "openai",
    "anthropic",
    "claude",
    "gemini",
    "openrouter",
    "chatbot",
    "chatbots",
    "function-calling",
    # Widened gate: borderline-but-legitimate ML/AI/data-science vocabulary that the
    # conservative core set missed, so genuine questions about classic ML, the maths
    # foundations, NLP/CV, RL, and the tooling ecosystem are no longer refused. Terms are
    # kept specific enough that they rarely fire on off-domain prose.
    "supervised",
    "unsupervised",
    "semi-supervised",
    "self-supervised",
    "contrastive",
    "encoder",
    "decoder",
    "seq2seq",
    "word2vec",
    "glove",
    "logits",
    "logit",
    "generative",
    "tensor",
    "tensors",
    "tensorflow",
    "pytorch",
    "keras",
    "jax",
    "sklearn",
    "scikit",
    "numpy",
    "pandas",
    "logistic",
    "lasso",
    "ridge",
    "knn",
    "naive",
    "ensemble",
    "bagging",
    "xgboost",
    "lightgbm",
    "adaboost",
    "kmeans",
    "dbscan",
    "pca",
    "svd",
    "tsne",
    "umap",
    "dimensionality",
    "anomaly",
    "outlier",
    "outliers",
    "imputation",
    "preprocessing",
    "standardization",
    "standardisation",
    "recommender",
    "recommendation",
    "perplexity",
    "bleu",
    "rouge",
    "auc",
    "roc",
    "ndcg",
    "calibration",
    "confusion",
    "likelihood",
    "posterior",
    "prior",
    "gaussian",
    "entropy",
    "covariance",
    "markov",
    "montecarlo",
    "bandit",
    "ppo",
    "dqn",
    "mdp",
    "q-learning",
    "adam",
    "adamw",
    "sgd",
    "rmsprop",
    "adagrad",
    "momentum",
    "scheduler",
    "pretrained",
    "pretraining",
    "nlp",
    "tokenization",
    "tokenisation",
    "lemmatization",
    "stemming",
    "corpus",
    "ngram",
    "vae",
    "multimodal",
    "yolo",
    "unet",
    "faiss",
    "pinecone",
    "weaviate",
    "milvus",
    "vectorstore",
    "reranker",
    "rerank",
    "reranking",
    "cuda",
    "onnx",
    "mlops",
    "ragas",
    "guardrails",
    "chunking",
}


@dataclass
class ValidationResult:
    """Outcome of input validation: `ok` plus a user-facing `reason` when rejected."""

    ok: bool
    reason: str = ""


def validate_input(text: str) -> ValidationResult:
    """Length/emptiness checks plus injection screening. Returns a user-facing reason."""
    stripped = text.strip()
    if not stripped:
        return ValidationResult(False, "Please type a question.")
    if len(stripped) > MAX_INPUT_CHARS:
        return ValidationResult(False, f"Question is too long (max {MAX_INPUT_CHARS} characters).")
    if detect_injection(stripped):
        return ValidationResult(
            False,
            "That message looks like a prompt-injection attempt, so I won't process it. "
            "Please ask a plain question about machine learning or AI.",
        )
    return ValidationResult(True)


def detect_injection(text: str) -> bool:
    """Return True if the text matches any known prompt-injection pattern."""
    return any(rx.search(text) for rx in _INJECTION_RE)


def has_domain_vocabulary(text: str) -> bool:
    """Return True if the text contains at least one ML/AI/DL domain keyword."""
    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", text.lower()))
    return bool(words & _DOMAIN_HINTS)


def is_on_domain(text: str, best_similarity: float, min_similarity: float) -> bool:
    """On-domain if retrieval found related material OR the wording is clearly ML/AI.

    The vocabulary fallback keeps tool-only requests (token counts, arXiv searches)
    answerable even when nothing relevant sits in the knowledge base.
    """
    return best_similarity >= min_similarity or has_domain_vocabulary(text)
