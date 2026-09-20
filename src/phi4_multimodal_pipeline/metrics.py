"""Corpus-level captioning metrics and two non-neural baselines, in pure Python.

`pipeline.py` keeps the per-caption plumbing check (`unigram_f1`); this module implements the three metrics a
captioning result is normally read by, over a set of records with several references each:

- **BLEU-4** (Papineni et al. 2002): corpus-level, clipped n-gram precision for n = 1..4 with geometric mean
  and brevity penalty against the closest reference length — the `coco-caption` convention.
- **ROUGE-L** (Lin 2004): per image the F-measure (beta = 1.2, as in `coco-caption`) of the longest common
  subsequence against the best-matching reference, averaged over images.
- **CIDEr-D** (Vedantam et al. 2015): TF-IDF-weighted n-gram cosine similarity for n = 1..4 with the
  Gaussian length penalty (sigma = 6) and clipping of candidate counts to the reference counts, averaged over
  references and n, scaled by 10. The document frequencies are computed over the references of the evaluated
  set, so a score is comparable only across systems evaluated on the same records (which is how it is used
  here: frozen vs. adapted vs. baselines on the same test split).

All three use COCO-style normalisation (`normalize_caption`: lower-case, punctuation removed). Two baselines a
fine-tuned model must beat: **constant caption** (the training caption that scores best against all other
training references — the corpus medoid, one string for every image) and **colour nearest neighbour** (the
caption of the training image whose 3x3 mean-colour grid is closest — a lookup that knows the image only
through 27 numbers).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import caption_tokens, unigram_f1

MAX_NGRAM = 4
ROUGE_BETA = 1.2
CIDER_SIGMA = 6.0
COLOUR_GRID = 3
METRIC_DEFINITIONS = {
    "bleu4": (
        "corpus-level BLEU-4: geometric mean of clipped 1..4-gram precisions against all references, times "
        "the brevity penalty against the closest reference length; in 0..1"
    ),
    "rouge_l": (
        "mean over images of the LCS-based F-measure (beta 1.2) against the best-matching reference; in 0..1"
    ),
    "cider_d": (
        "mean over images of the CIDEr-D consensus score: TF-IDF-weighted 1..4-gram cosine similarity to the "
        "references with a Gaussian length penalty (sigma 6), clipped counts, scaled by 10; document "
        "frequencies from the references of the evaluated set; typically 0..~1.5 on VizWiz"
    ),
    "unigram_f1": "mean over images of the bag-of-words F1 against the best-matching reference; in 0..1",
}


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _check(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> None:
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions but {len(references)} reference lists")
    if not predictions:
        raise ValueError("no predictions to score")
    for index, refs in enumerate(references):
        if isinstance(refs, str) or not refs:
            raise ValueError(f"references[{index}] must be a non-empty list of captions")


def bleu4(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """Corpus-level BLEU-4 with multi-reference clipping and the closest-length brevity penalty."""
    _check(predictions, references)
    matched = [0] * MAX_NGRAM
    total = [0] * MAX_NGRAM
    hyp_len = ref_len = 0
    for prediction, refs in zip(predictions, references, strict=True):
        hyp = caption_tokens(prediction)
        ref_tokens = [caption_tokens(r) for r in refs]
        hyp_len += len(hyp)
        ref_len += min((abs(len(r) - len(hyp)), len(r)) for r in ref_tokens)[1]
        for n in range(1, MAX_NGRAM + 1):
            counts = _ngrams(hyp, n)
            max_ref: Counter[tuple[str, ...]] = Counter()
            for r in ref_tokens:
                for gram, count in _ngrams(r, n).items():
                    max_ref[gram] = max(max_ref[gram], count)
            matched[n - 1] += sum(min(count, max_ref[gram]) for gram, count in counts.items())
            total[n - 1] += max(len(hyp) - n + 1, 0)
    if any(m == 0 for m in matched) or hyp_len == 0:
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matched, total, strict=True)) / MAX_NGRAM
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return brevity * math.exp(log_precision)


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(previous[j] + 1 if token == other else max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, references: Sequence[str]) -> float:
    """LCS F-measure (beta 1.2) against the best-matching reference."""
    hyp = caption_tokens(prediction)
    best = 0.0
    for reference in references:
        ref = caption_tokens(reference)
        lcs = _lcs(hyp, ref)
        if lcs == 0:
            continue
        precision, recall = lcs / len(hyp), lcs / len(ref)
        best = max(best, (1 + ROUGE_BETA**2) * precision * recall / (recall + ROUGE_BETA**2 * precision))
    return best


def _document_frequencies(references: Sequence[Sequence[str]]) -> list[Counter[tuple[str, ...]]]:
    """Per n, the number of images whose references contain each n-gram."""
    dfs: list[Counter[tuple[str, ...]]] = [Counter() for _ in range(MAX_NGRAM)]
    for refs in references:
        for n in range(1, MAX_NGRAM + 1):
            seen: set[tuple[str, ...]] = set()
            for r in refs:
                seen.update(_ngrams(caption_tokens(r), n))
            for gram in seen:
                dfs[n - 1][gram] += 1
    return dfs


def _tfidf(
    tokens: Sequence[str], n: int, df: Counter[tuple[str, ...]], log_images: float
) -> tuple[dict[tuple[str, ...], float], float]:
    counts = _ngrams(tokens, n)
    vector = {gram: count * (log_images - math.log(max(df[gram], 1.0))) for gram, count in counts.items()}
    return vector, math.sqrt(sum(v * v for v in vector.values()))


class _CiderIndex:
    """Document frequencies and reference TF-IDF vectors of one evaluated set, computed once."""

    def __init__(self, references: Sequence[Sequence[str]]) -> None:
        self.dfs = _document_frequencies(references)
        self.log_images = math.log(len(references))
        self.refs: list[list[tuple[list[str], list[tuple[dict[tuple[str, ...], float], float]]]]] = []
        for refs in references:
            entry = []
            for r in refs:
                tokens = caption_tokens(r)
                entry.append((tokens, [self._vector(tokens, n) for n in range(1, MAX_NGRAM + 1)]))
            self.refs.append(entry)

    def _vector(self, tokens: Sequence[str], n: int) -> tuple[dict[tuple[str, ...], float], float]:
        return _tfidf(tokens, n, self.dfs[n - 1], self.log_images)

    def score(self, prediction: str, image_index: int) -> float:
        hyp = caption_tokens(prediction)
        hyp_vectors = [self._vector(hyp, n) for n in range(1, MAX_NGRAM + 1)]
        refs = self.refs[image_index]
        total = 0.0
        for ref_tokens, ref_vectors in refs:
            penalty = math.exp(-((len(hyp) - len(ref_tokens)) ** 2) / (2 * CIDER_SIGMA**2))
            for (hyp_vec, hyp_norm), (ref_vec, ref_norm) in zip(hyp_vectors, ref_vectors, strict=True):
                if hyp_norm == 0 or ref_norm == 0:
                    continue
                dot = sum(min(hyp_vec[g], ref_vec[g]) * ref_vec[g] for g in hyp_vec if g in ref_vec)
                total += dot / (hyp_norm * ref_norm) * penalty
        return 10.0 * total / (MAX_NGRAM * len(refs))


def cider_d(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> list[float]:
    """Per-image CIDEr-D scores with document frequencies from `references` (Vedantam et al. 2015,
    the `coco-caption` implementation)."""
    _check(predictions, references)
    index = _CiderIndex(references)
    return [index.score(p, i) for i, p in enumerate(predictions)]


def caption_metrics(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> dict[str, Any]:
    """BLEU-4, ROUGE-L, CIDEr-D and unigram F1 over parallel predictions and reference lists."""
    _check(predictions, references)
    pairs = list(zip(predictions, references, strict=True))
    cider = cider_d(predictions, references)
    return {
        "n": len(predictions),
        "bleu4": bleu4(predictions, references),
        "rouge_l": sum(rouge_l(p, r) for p, r in pairs) / len(pairs),
        "cider_d": sum(cider) / len(cider),
        "unigram_f1": sum(unigram_f1(p, r) for p, r in pairs) / len(pairs),
        "empty_rate": sum(1 for p in predictions if not caption_tokens(p)) / len(predictions),
        "mean_words": sum(len(caption_tokens(p)) for p in predictions) / len(predictions),
        "definitions": dict(METRIC_DEFINITIONS),
    }


def _references(records: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    return [[str(c) for c in r["captions"]] for r in records]


def medoid_caption(train: Sequence[Mapping[str, Any]]) -> str:
    """The training caption with the highest mean CIDEr-D against every other training image's references."""
    if not train:
        raise ValueError("the constant-caption baseline needs training records")
    candidates = sorted({str(c) for r in train for c in r["captions"]})
    index = _CiderIndex(_references(train))
    best, best_score = candidates[0], -1.0
    for candidate in candidates:
        score = sum(index.score(candidate, i) for i in range(len(train))) / len(train)
        if score > best_score:
            best, best_score = candidate, score
    return best


def constant_caption_baseline(
    train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The same caption (the training medoid) for every image."""
    caption = medoid_caption(train)
    result = caption_metrics([caption] * len(records), _references(records))
    result["baseline"] = f"constant caption {caption!r}"
    return result


def colour_signature(image: str | Path | Image.Image, *, grid: int = COLOUR_GRID) -> list[float]:
    """Mean RGB of each cell of a `grid` x `grid` partition of the image, in 0..1 (27 numbers by default)."""
    handle = image if isinstance(image, Image.Image) else Image.open(image)
    with handle:
        small = handle.convert("RGB").resize((grid * 8, grid * 8), Image.BILINEAR)
        pixels = list(small.getdata())
    out: list[float] = []
    for row in range(grid):
        for col in range(grid):
            cell = [pixels[(row * 8 + y) * grid * 8 + col * 8 + x] for y in range(8) for x in range(8)]
            out.extend(sum(p[channel] for p in cell) / (64 * 255.0) for channel in range(3))
    return out


def colour_neighbour_baseline(
    train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Caption every image with the first reference of the training image whose colour signature is
    closest (Euclidean distance over the 3x3 mean-colour grid; no model)."""
    if not train:
        raise ValueError("the colour-neighbour baseline needs training records")
    signatures = [(colour_signature(r["image"]), str(r["captions"][0])) for r in train]
    predictions = []
    for record in records:
        query = colour_signature(record["image"])
        predictions.append(
            min(signatures, key=lambda s: sum((a - b) ** 2 for a, b in zip(s[0], query, strict=True)))[1]
        )
    result = caption_metrics(predictions, _references(records))
    result["baseline"] = f"colour nearest neighbour ({len(train)} training images)"
    return result
