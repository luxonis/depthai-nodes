"""Host parser for raw Code 128 CTC neural-network tensors."""

from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.code128 import Code128Result

from .base_parser import BaseParser
from .utils.classification_sequence import compute_classification_sequence_scores
from .utils.code128 import (
    beam_search_code128,
    codeword_classes,
    greedy_decode_code128,
    validate_decoding_options,
)


class Code128CTCParser(BaseParser):
    """Decode full codewords (blank 0, codeword n at index n+1).

    The NNArchive head supplies ``outputs``, ``classes``, ``is_softmax`` and
    ``target_mode='codewords'``. Optional decoding_method, beam_width, top_k,
    token_prune and require_valid_checksum may be overridden with configure().
    """

    @property
    def input(self):
        return self._input

    @property
    def out(self):
        return self._out

    def build(self, head_config: dict[str, Any]) -> "Code128CTCParser":
        outputs = head_config.get("outputs", [])
        if len(outputs) != 1:
            raise ValueError("Code128CTCParser requires exactly one output")
        if head_config.get("target_mode") != "codewords":
            raise ValueError("Code128CTCParser requires target_mode='codewords'")
        if (
            head_config.get("classes") != codeword_classes()
            or head_config.get("n_classes") != 108
        ):
            raise ValueError(
                "Code128CTCParser requires the 108-class Code 128 codeword alphabet"
            )
        if head_config.get("ignored_indexes", [0]) != [0] or not head_config.get(
            "remove_duplicates", True
        ):
            raise ValueError(
                "Code128CTCParser requires CTC blank 0 and duplicate collapsing"
            )
        self.output_layer_name = outputs[0]
        self.is_softmax = head_config.get("is_softmax", True)
        if not isinstance(self.is_softmax, bool):
            raise ValueError("is_softmax must be boolean")
        self.output_layout = head_config.get("output_layout", "NTC")
        if self.output_layout not in ("NTC", "NCT", "TC", "CT"):
            raise ValueError("output_layout must be NTC, NCT, TC or CT")
        self.decoding_method = "greedy"
        self.beam_width = 10
        self.top_k = 5
        self.token_prune = None
        self.require_valid_checksum = True
        self.configure(**{k: head_config[k] for k in self.OPTIONS if k in head_config})
        return self

    OPTIONS = {
        "decoding_method",
        "beam_width",
        "top_k",
        "token_prune",
        "require_valid_checksum",
    }

    def configure(self, **overrides) -> "Code128CTCParser":
        """Override archive decoding defaults before starting the pipeline."""
        unknown = overrides.keys() - self.OPTIONS
        if unknown:
            raise ValueError(f"Unknown Code 128 parser options: {sorted(unknown)}")
        options = {
            name: overrides.get(name, getattr(self, name)) for name in self.OPTIONS
        }
        validate_decoding_options(**options)
        for name, value in options.items():
            setattr(self, name, value)
        return self

    def compute(self, scores: np.ndarray) -> Code128Result:
        scores = np.asarray(scores, dtype=np.float32)
        if self.output_layout in ("NCT", "CT"):
            scores = scores.swapaxes(-1, -2)
        scores = compute_classification_sequence_scores(
            scores, is_softmax=self.is_softmax
        )
        if self.decoding_method == "greedy":
            candidates = [
                greedy_decode_code128(
                    scores, require_valid_checksum=self.require_valid_checksum
                )
            ]
        else:
            candidates = beam_search_code128(
                scores,
                beam_width=self.beam_width,
                top_k=self.top_k,
                token_prune=self.token_prune,
                require_valid_checksum=self.require_valid_checksum,
            )
        return Code128Result(candidates)

    def process(self, output: dai.NNData) -> Code128Result:
        result = self.compute(output.getTensor(self.output_layer_name, dequantize=True))
        result.setSequenceNum(output.getSequenceNum())
        result.setTimestamp(output.getTimestamp())
        result.setTimestampDevice(output.getTimestampDevice())
        return result

    def run(self) -> None:
        while self.isRunning():
            try:
                output = self.input.get()
            except dai.MessageQueue.QueueException:
                break
            self.out.send(self.process(output))
