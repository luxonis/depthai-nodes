def compute_embeddings_output(output):
    """Pass an embedding payload through unchanged.

    Args:
        output: Embedding payload, typically a ``dai.NNData`` message.

    Returns:
        The same object passed as ``output``; no copy or normalization is performed.
    """
    return output
