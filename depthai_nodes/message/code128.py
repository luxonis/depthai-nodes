"""Structured output of the Code 128 CTC parser."""

import depthai as dai


class Code128Result(dai.Buffer):
    """One result per crop, including failed predictions and ranked candidates.

    ``score`` is a CTC log probability, not a calibrated confidence. ``text``
    is None on failure. ``candidates`` retain codewords and validation errors.
    """

    def __init__(self, candidates):
        super().__init__()
        self.candidates = tuple(candidates)
        best = max(self.candidates, key=lambda c: (c.valid, c.score))
        self.text: str | None = best.text if best.valid else None
        self.codewords: tuple[int, ...] = best.codewords
        self.score: float = best.score
        self.valid: bool = best.valid
        self.checksum_valid: bool = best.checksum_valid
        self.error: str | None = best.error
