"""NumPy-only CTC decoding and validation of full Code 128 symbols."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Code128Candidate:
    codewords: tuple[int, ...]
    text: str
    score: float  # CTC sequence log probability; not a calibrated confidence.
    valid: bool
    checksum_valid: bool
    error: str | None = None


def codeword_classes() -> list[str]:
    names = {
        99: "TO_C",
        100: "TO_B",
        101: "TO_A",
        103: "START_A",
        104: "START_B",
        105: "START_C",
        106: "STOP",
    }
    return [""] + [names.get(i, str(i)) for i in range(107)]


def decode_codewords(
    codewords: tuple[int, ...], *, require_valid_checksum: bool = True
) -> tuple[str, bool]:
    """Interpret code sets A/B/C, SHIFT and FNC1/FNC4; reject unsupported controls.

    A leading FNC1 marks GS1 and is omitted from text; subsequent FNC1s become
    ASCII GS. FNC2/FNC3 require application-level semantics and are rejected.
    """
    if (
        len(codewords) < 4
        or codewords[0] not in (103, 104, 105)
        or codewords[-1] != 106
    ):
        raise ValueError("Invalid start/stop or symbol length")
    if any(not 0 <= c <= 102 for c in codewords[1:-1]):
        raise ValueError("Invalid data or checksum codeword")
    expected = (
        codewords[0] + sum(i * c for i, c in enumerate(codewords[1:-2], 1))
    ) % 103
    checksum_valid = codewords[-2] == expected
    if require_valid_checksum and not checksum_valid:
        raise ValueError("Checksum mismatch")
    codeset = {103: "A", 104: "B", 105: "C"}[codewords[0]]
    text: list[str] = []
    upper = False
    payload = codewords[1:-2]
    i = 0
    while i < len(payload):
        value = payload[i]
        i += 1
        if value == 102:
            if i != 1:
                text.append("\x1d")
            continue
        if codeset == "C":
            if value <= 99:
                text.append(f"{value:02d}")
            else:
                codeset = {100: "B", 101: "A"}[value]
            continue
        if value <= 95:
            char = value + 32 if codeset == "B" or value < 64 else value - 64
            text.append(chr(char + (128 if upper else 0)))
        elif value == 99:
            codeset = "C"
        elif value == (101 if codeset == "B" else 100):
            codeset = "A" if codeset == "B" else "B"
        elif value == 98:
            if i == len(payload) or payload[i] > 95:
                raise ValueError("SHIFT must precede a data codeword")
            shifted = "A" if codeset == "B" else "B"
            value = payload[i]
            i += 1
            char = value + 32 if shifted == "B" or value < 64 else value - 64
            text.append(chr(char + (128 if upper else 0)))
        elif value == (100 if codeset == "B" else 101):
            if i < len(payload) and payload[i] == value:
                upper = not upper
                i += 1
            else:
                if i == len(payload) or payload[i] > 95:
                    raise ValueError("FNC4 must precede data or another FNC4")
                value = payload[i]
                i += 1
                char = value + 32 if codeset == "B" or value < 64 else value - 64
                text.append(chr(char + (0 if upper else 128)))
        else:
            raise ValueError("Unsupported FNC2/FNC3 control")
    return "".join(text), checksum_valid


def _candidate(
    tokens: tuple[int, ...], score: float, require_valid_checksum: bool
) -> Code128Candidate:
    words = tuple(token - 1 for token in tokens)
    try:
        text, checksum = decode_codewords(
            words, require_valid_checksum=require_valid_checksum
        )
        return Code128Candidate(words, text, score, True, checksum)
    except ValueError as exc:
        try:
            _, checksum = decode_codewords(words, require_valid_checksum=False)
        except ValueError:
            checksum = False
        return Code128Candidate(
            words, f"INVALID {list(words)}", score, False, checksum, str(exc)
        )


def validate_decoding_options(
    decoding_method: str,
    beam_width: int,
    top_k: int,
    token_prune: int | None,
    require_valid_checksum: bool,
) -> None:
    if decoding_method not in ("greedy", "beam_search"):
        raise ValueError("decoding_method must be greedy or beam_search")
    for name, value in (("beam_width", beam_width), ("top_k", top_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if top_k > beam_width:
        raise ValueError("top_k cannot exceed beam_width")
    if token_prune is not None and (
        isinstance(token_prune, bool)
        or not isinstance(token_prune, int)
        or not 1 <= token_prune <= 108
    ):
        raise ValueError("token_prune must be an integer in [1, 108] or None")
    if not isinstance(require_valid_checksum, bool):
        raise ValueError("require_valid_checksum must be boolean")


def _probabilities(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 108 or scores.shape[0] == 0:
        raise ValueError("Expected nonempty (steps, 108) Code 128 probabilities")
    if (
        not np.all(np.isfinite(scores))
        or np.any(scores < 0)
        or np.any(scores.sum(axis=1) <= 0)
    ):
        raise ValueError(
            "Probabilities must be finite, nonnegative and have positive row sums"
        )
    return scores / scores.sum(axis=1, keepdims=True)


def greedy_decode_code128(
    scores: np.ndarray, *, require_valid_checksum: bool = True
) -> Code128Candidate:
    probabilities = _probabilities(scores)
    ids = probabilities.argmax(axis=1)
    # Collapse repeats before removing blanks: x, blank, x is two tokens.
    tokens = tuple(
        int(t) for i, t in enumerate(ids) if t != 0 and (i == 0 or t != ids[i - 1])
    )
    score = float(
        np.log(
            np.maximum(probabilities[np.arange(len(ids)), ids], np.finfo(float).tiny)
        ).sum()
    )
    return _candidate(tokens, score, require_valid_checksum)


def beam_search_code128(
    scores: np.ndarray,
    *,
    beam_width: int = 10,
    top_k: int = 5,
    token_prune: int | None = None,
    require_valid_checksum: bool = True,
) -> list[Code128Candidate]:
    """CTC prefix beam search summing blank/nonblank alignment probabilities."""
    validate_decoding_options(
        "beam_search", beam_width, top_k, token_prune, require_valid_checksum
    )
    probabilities = _probabilities(scores)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
    for row in probabilities:
        logs = np.full_like(row, -math.inf)
        np.log(row, out=logs, where=row > 0)
        ids = np.arange(108) if token_prune is None else np.argsort(row)[-token_prune:]
        ids = np.unique(np.append(ids, 0))
        following: dict[tuple[int, ...], list[float]] = {}

        def add(
            prefix: tuple[int, ...], channel: int, value: float, target=following
        ) -> None:
            if value == -math.inf:
                return
            entry = target.setdefault(prefix, [-math.inf, -math.inf])
            entry[channel] = float(np.logaddexp(entry[channel], value))

        for prefix, (blank, nonblank) in beams.items():
            total = float(np.logaddexp(blank, nonblank))
            for raw_token in ids:
                token = int(raw_token)
                p = float(logs[token])
                if p == -math.inf:
                    continue
                if token == 0:
                    add(prefix, 0, total + p)
                elif prefix and token == prefix[-1]:
                    add(prefix, 1, nonblank + p)
                    add(prefix + (token,), 1, blank + p)
                else:
                    add(prefix + (token,), 1, total + p)
        ranked = sorted(
            following.items(),
            key=lambda pair: float(np.logaddexp(*pair[1])),
            reverse=True,
        )
        beams = {
            prefix: (values[0], values[1]) for prefix, values in ranked[:beam_width]
        }
    ranked = sorted(
        beams.items(), key=lambda pair: float(np.logaddexp(*pair[1])), reverse=True
    )
    return [
        _candidate(prefix, float(np.logaddexp(*values)), require_valid_checksum)
        for prefix, values in ranked[:top_k]
    ]


def select_best_code128_candidate(
    candidates: list[Code128Candidate],
) -> Code128Candidate:
    if not candidates:
        raise ValueError("No CTC candidates")
    return max(candidates, key=lambda c: (c.valid, c.score))


def normalize_candidate_scores(candidates: list[Code128Candidate]) -> np.ndarray:
    """Relative probabilities among retained candidates, not absolute confidence."""
    if not candidates:
        return np.empty(0, dtype=np.float32)
    logs = np.asarray([c.score for c in candidates], dtype=np.float64)
    values = np.exp(logs - logs.max())
    return (values / values.sum()).astype(np.float32)
