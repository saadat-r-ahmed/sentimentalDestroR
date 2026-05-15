"""All evaluation metrics used in the destroR pipeline."""
from __future__ import annotations

from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Attack success rate
# ---------------------------------------------------------------------------

def attack_success_rate(results: list) -> float:
    """ASR = fraction of attackable examples where attack succeeded."""
    attackable = [r for r in results if not r.meta.get("skipped")]
    if not attackable:
        return 0.0
    return sum(r.success for r in attackable) / len(attackable)


# ---------------------------------------------------------------------------
# Perturbation rate (token-level)
# ---------------------------------------------------------------------------

def perturbation_rate(original: str, adversarial: str) -> float:
    """Fraction of tokens that changed between original and adversarial text.

    Uses whitespace tokenization as a fast approximation; callers that need
    Bangla-aware tokenization should pre-tokenize and pass the joined string.
    """
    orig_toks = original.split()
    adv_toks = adversarial.split()
    if not orig_toks:
        return 0.0
    n_changed = sum(a != b for a, b in zip(orig_toks, adv_toks))
    n_changed += abs(len(orig_toks) - len(adv_toks))
    return round(n_changed / len(orig_toks) * 100, 2)


# ---------------------------------------------------------------------------
# Semantic similarity — LaBSE cosine
# ---------------------------------------------------------------------------

_labse_model = None


def _get_labse():
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        _labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    return _labse_model


def semantic_similarity(original: str, adversarial: str) -> float:
    """LaBSE cosine similarity between original and adversarial text."""
    model = _get_labse()
    embs = model.encode([original, adversarial], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def batch_semantic_similarity(originals: list[str], adversarials: list[str]) -> list[float]:
    model = _get_labse()
    all_texts = originals + adversarials
    embs = model.encode(all_texts, normalize_embeddings=True, batch_size=64)
    n = len(originals)
    return [float(np.dot(embs[i], embs[n + i])) for i in range(n)]


# ---------------------------------------------------------------------------
# Perplexity delta — uses a held-out Bangla LM
# ---------------------------------------------------------------------------

_ppl_model = None
_ppl_tokenizer = None


def _get_ppl_model():
    global _ppl_model, _ppl_tokenizer
    if _ppl_model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # BanglaGPT or any autoregressive Bangla LM
        model_name = "csebuetnlp/banglat5"
        _ppl_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _ppl_model = AutoModelForCausalLM.from_pretrained(model_name)
        _ppl_model.eval()
    return _ppl_model, _ppl_tokenizer


def _compute_perplexity(text: str) -> float:
    import torch
    model, tokenizer = _get_ppl_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return float(torch.exp(loss))


def perplexity_delta(original: str, adversarial: str) -> float:
    """PPL(adversarial) - PPL(original). Positive means more perplexed."""
    return round(_compute_perplexity(adversarial) - _compute_perplexity(original), 4)


# ---------------------------------------------------------------------------
# Translation-reference metrics (BLEU, chrF)
# ---------------------------------------------------------------------------

def bleu_score(reference: str, hypothesis: str) -> float:
    from sacrebleu.metrics import BLEU
    bleu = BLEU(effective_order=True)
    return bleu.sentence_score(hypothesis, [reference]).score


def chrf_score(reference: str, hypothesis: str) -> float:
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    return chrf.sentence_score(hypothesis, [reference]).score
