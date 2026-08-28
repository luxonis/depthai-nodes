from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

CODE128_TO_C = 99
CODE128_TO_B = 100
CODE128_TO_A = 101
CODE128_START_A = 103
CODE128_START_B = 104
CODE128_START_C = 105
CODE128_STOP = 106

CTC_BLANK_INDEX = 0
NUM_CODEWORDS = 107

_CODESET_A = "A"
_CODESET_B = "B"
_CODESET_C = "C"


@dataclass(frozen=True)
class Code128BeamCandidate:
    codewords: tuple[int, ...]
    score: float
    text: str
    valid: bool
    checksum_valid: bool


def _token_name(codeword: int) -> str:
    if codeword == CODE128_TO_C:
        return "TO_C"
    if codeword == CODE128_TO_B:
        return "TO_B"
    if codeword == CODE128_TO_A:
        return "TO_A"
    if codeword == CODE128_START_A:
        return "START_A"
    if codeword == CODE128_START_B:
        return "START_B"
    if codeword == CODE128_START_C:
        return "START_C"
    if codeword == CODE128_STOP:
        return "STOP"
    return str(codeword)


def codeword_names() -> list[str]:
    return [""] + [_token_name(codeword) for codeword in range(NUM_CODEWORDS)]


def parse_codeword_sequence(
    value: str | bytes | list[int] | tuple[int, ...] | np.ndarray,
) -> list[int]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return parse_codeword_sequence(value.item())
        if value.dtype.kind in {"i", "u"}:
            return [int(codeword) for codeword in value.reshape(-1).tolist()]
        if value.size == 1:
            scalar = value.reshape(-1)[0]
            if hasattr(scalar, "item"):
                scalar = scalar.item()
            return parse_codeword_sequence(scalar)
        raise TypeError(
            "Unsupported ndarray target for Code128 codewords. Expected a "
            "single serialized value or an integer array."
        )
    if isinstance(value, list):
        return [int(codeword) for codeword in value]
    if isinstance(value, tuple):
        return [int(codeword) for codeword in value]
    if not isinstance(value, str):
        raise TypeError(f"Unsupported codeword sequence type: {type(value)!r}")

    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Serialized codeword sequence must decode to a list.")
    return [int(codeword) for codeword in parsed]


def _compute_checksum(codewords_without_checksum: list[int]) -> int:
    checksum_terms = [codewords_without_checksum[0]]
    for index, value in enumerate(codewords_without_checksum[1:], start=1):
        checksum_terms.append(index * value)
    return sum(checksum_terms) % 103


def _start_codeset_from_codeword(codeword: int) -> str:
    if codeword == CODE128_START_A:
        return _CODESET_A
    if codeword == CODE128_START_B:
        return _CODESET_B
    if codeword == CODE128_START_C:
        return _CODESET_C
    raise ValueError(f"Codeword {codeword} is not a valid Code128 start token.")


def _decode_char_in_codeset(codeword: int, codeset: str) -> str:
    if not 0 <= codeword <= 95:
        raise ValueError(f"Codeword {codeword} is outside the data range 0..95.")
    if codeset == _CODESET_A:
        return chr(codeword + 32) if codeword <= 63 else chr(codeword - 64)
    if codeset == _CODESET_B:
        return chr(codeword + 32)
    raise ValueError(f"Unsupported codeset {codeset!r}.")


def code128_codewords_to_text(codewords: list[int] | tuple[int, ...]) -> str:
    if len(codewords) < 4:
        raise ValueError("Sequence is too short to be a valid Code128 read.")
    if codewords[-1] != CODE128_STOP:
        raise ValueError("Sequence does not end with STOP.")

    payload_codewords = list(codewords[:-2])
    checksum = codewords[-2]
    codeset = _start_codeset_from_codeword(payload_codewords[0])

    expected_checksum = _compute_checksum(payload_codewords)
    if checksum != expected_checksum:
        raise ValueError("Checksum mismatch.")

    decoded: list[str] = []
    for codeword in payload_codewords[1:]:
        if codeset == _CODESET_C:
            if 0 <= codeword <= 99:
                decoded.append(f"{codeword:02d}")
                continue
            if codeword == CODE128_TO_A:
                codeset = _CODESET_A
                continue
            if codeword == CODE128_TO_B:
                codeset = _CODESET_B
                continue
            raise ValueError(
                f"Codeword {codeword} is invalid while decoding Code Set C."
            )

        if 0 <= codeword <= 95:
            decoded.append(_decode_char_in_codeset(codeword, codeset))
            continue
        if codeword == CODE128_TO_A:
            codeset = _CODESET_A
            continue
        if codeword == CODE128_TO_B:
            codeset = _CODESET_B
            continue
        if codeword == CODE128_TO_C:
            codeset = _CODESET_C
            continue
        raise ValueError(
            f"Codeword {codeword} is invalid while decoding Code Set {codeset}."
        )

    return "".join(decoded)


def _describe_codeword_sequence_body(
    codewords: list[int] | tuple[int, ...],
) -> list[str]:
    if not codewords:
        return []

    parts = [_token_name(codewords[0])]
    try:
        codeset = _start_codeset_from_codeword(codewords[0])
    except ValueError:
        parts.extend(_token_name(codeword) for codeword in codewords[1:])
        return parts

    for codeword in codewords[1:]:
        if codeset == _CODESET_C:
            if 0 <= codeword <= 99:
                parts.append(str(codeword))
                continue
            if codeword == CODE128_TO_A:
                parts.append("TO_A")
                codeset = _CODESET_A
                continue
            if codeword == CODE128_TO_B:
                parts.append("TO_B")
                codeset = _CODESET_B
                continue
            if codeword == CODE128_STOP:
                parts.append("STOP")
                continue
            parts.append(str(codeword))
            continue

        if 0 <= codeword <= 95:
            parts.append(str(codeword))
            continue
        if codeword == CODE128_TO_A:
            parts.append("TO_A")
            codeset = _CODESET_A
            continue
        if codeword == CODE128_TO_B:
            parts.append("TO_B")
            codeset = _CODESET_B
            continue
        if codeword == CODE128_TO_C:
            parts.append("TO_C")
            codeset = _CODESET_C
            continue
        if codeword == CODE128_STOP:
            parts.append("STOP")
            continue
        parts.append(str(codeword))

    return parts


def describe_codeword_sequence(codewords: list[int] | tuple[int, ...]) -> str:
    return "[" + ", ".join(_describe_codeword_sequence_body(codewords)) + "]"


def _checksum_matches(codewords: tuple[int, ...]) -> bool:
    if len(codewords) < 4 or codewords[-1] != CODE128_STOP:
        return False
    try:
        _start_codeset_from_codeword(codewords[0])
    except ValueError:
        return False
    payload = list(codewords[:-2])
    checksum = codewords[-2]
    return checksum == _compute_checksum(payload)


def _candidate_text(codewords: tuple[int, ...]) -> tuple[str, bool]:
    try:
        return code128_codewords_to_text(codewords), True
    except ValueError:
        return f"INVALID {describe_codeword_sequence(codewords)}", False


def _top_token_indices(
    timestep: np.ndarray,
    prune_k: int,
) -> list[int]:
    if prune_k >= len(timestep):
        return np.argsort(-timestep, kind="stable").tolist()
    indices = np.argpartition(-timestep, prune_k - 1)[:prune_k]
    ranked = indices[np.argsort(-timestep[indices], kind="stable")]
    return ranked.tolist()


def beam_search_code128(
    scores: np.ndarray,
    *,
    beam_width: int = 10,
    top_k: int = 5,
    token_prune: int | None = None,
) -> list[Code128BeamCandidate]:
    log_probs = np.log(np.clip(np.asarray(scores, dtype=np.float64), 1e-12, 1.0))
    blank = CTC_BLANK_INDEX
    n_classes = log_probs.shape[-1]
    prune_k = token_prune or min(n_classes, max(beam_width * 3, 8))
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -np.inf)}

    for timestep in log_probs:
        next_beams: dict[tuple[int, ...], list[float]] = {}
        top_indices = _top_token_indices(timestep, prune_k)

        for prefix, (p_blank, p_nonblank) in beams.items():
            prefix_total = float(np.logaddexp(p_blank, p_nonblank))
            for token_id in top_indices:
                probability = float(timestep[token_id])
                if token_id == blank:
                    entry = next_beams.setdefault(prefix, [-np.inf, -np.inf])
                    entry[0] = float(
                        np.logaddexp(entry[0], prefix_total + probability)
                    )
                    continue

                if prefix and token_id == prefix[-1]:
                    repeat_entry = next_beams.setdefault(prefix, [-np.inf, -np.inf])
                    repeat_entry[1] = float(
                        np.logaddexp(repeat_entry[1], p_nonblank + probability)
                    )

                    extended_prefix = prefix + (token_id,)
                    extended_entry = next_beams.setdefault(
                        extended_prefix, [-np.inf, -np.inf]
                    )
                    extended_entry[1] = float(
                        np.logaddexp(extended_entry[1], p_blank + probability)
                    )
                else:
                    extended_prefix = prefix + (token_id,)
                    extended_entry = next_beams.setdefault(
                        extended_prefix, [-np.inf, -np.inf]
                    )
                    extended_entry[1] = float(
                        np.logaddexp(
                            extended_entry[1], prefix_total + probability
                        )
                    )

        ranked = sorted(
            (
                (prefix, values[0], values[1])
                for prefix, values in next_beams.items()
            ),
            key=lambda item: float(np.logaddexp(item[1], item[2])),
            reverse=True,
        )[:beam_width]
        beams = {
            prefix: (p_blank, p_nonblank)
            for prefix, p_blank, p_nonblank in ranked
        }

    ranked_prefixes = sorted(
        beams.items(),
        key=lambda item: float(np.logaddexp(item[1][0], item[1][1])),
        reverse=True,
    )[:top_k]

    candidates: list[Code128BeamCandidate] = []
    for prefix, (p_blank, p_nonblank) in ranked_prefixes:
        codewords = tuple(token_id - 1 for token_id in prefix)
        score = float(np.logaddexp(p_blank, p_nonblank))
        text, valid = _candidate_text(codewords)
        candidates.append(
            Code128BeamCandidate(
                codewords=codewords,
                score=score,
                text=text,
                valid=valid,
                checksum_valid=_checksum_matches(codewords),
            )
        )
    return candidates


def select_best_code128_candidate(
    candidates: list[Code128BeamCandidate],
    *,
    prefer_valid_checksum: bool = True,
) -> Code128BeamCandidate:
    if not candidates:
        return Code128BeamCandidate(
            codewords=(),
            score=float("-inf"),
            text="INVALID []",
            valid=False,
            checksum_valid=False,
        )

    if prefer_valid_checksum:
        valid_candidates = [candidate for candidate in candidates if candidate.valid]
        checksum_candidates = [
            candidate for candidate in candidates if candidate.checksum_valid
        ]
        if valid_candidates:
            return valid_candidates[0]
        if checksum_candidates:
            return checksum_candidates[0]
    return candidates[0]


def normalize_candidate_scores(
    candidates: list[Code128BeamCandidate],
) -> np.ndarray:
    if not candidates:
        return np.array([1.0], dtype=np.float32)

    log_scores = np.array([candidate.score for candidate in candidates], dtype=np.float64)
    log_scores -= np.max(log_scores)
    scores = np.exp(log_scores)
    total = np.sum(scores)
    if total <= 0 or not np.isfinite(total):
        return np.full(len(candidates), 1.0 / len(candidates), dtype=np.float32)
    return (scores / total).astype(np.float32)
