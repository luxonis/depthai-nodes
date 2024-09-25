from typing import Dict, List, Union, Optional

import depthai as dai


class Parser(dai.node.ThreadedHostNode):
    """Base class for neural network output parsers.

    This class serves as a foundation for specific parser implementations used to postprocess the outputs of neural network models.
    Each parser is attached to a model "head" that governs the parsing process as it contains all the necessary information for the parser to function correctly.
    Subclasses should implement the `run` method to define the parsing logic.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    head_config : Dict
        A dictionary containing configuration details relevant to the parser, including parameters and settings required for output parsing.
    """

    def __init__(self):
        super().__init__()
        self.input = self.createInput()
        self.out = self.createOutput()

        self.head_config: Dict = {}

    def build(self, nn_archive: dai.NNArchive, head_name: str = None):
        """Sets the head configuration for the specified head.

        Attributes
        ----------
        nn_archive: dai.NNArchive
            NN Archive of the model.
        head_name : str
            The name of the head to use. If multiple heads are available, the name must be specified.

        Returns
        -------
        Parser
            Returns the parser object with the head configuration set.
        """

        if not isinstance(nn_archive, dai.NNArchive):
            raise ValueError(
                f"Provided heads must be of type depthai.NNArchive not {type(nn_archive)}."
            )

        try:
            heads = nn_archive.getConfig().getConfigV1().model.heads
        except:
            raise ValueError(
                "Only the NN Archives of version V1 are supported."
            )

        if len(heads) == 0:
            raise ValueError("No heads defined in the NN Archive.")
        elif len(heads) == 1:
            head = heads[0]
        else:
            if head_name:
                head_candidates = [
                    head
                    for head in heads
                    if head.metadata.extraParams["name"] == head_name
                ]
                if len(head_candidates) == 0:
                    raise ValueError(
                        f"No head with name {head_name} specified in NN Archive."
                    )
                if len(head_candidates) > 1:
                    raise ValueError(
                        f"Multiple heads with name {head_name} found in NN Archive, please specify a unique name."
                    )
                head = head_candidates[0]
            else:
                current_parser = self.__class__.__name__
                parser_names_in_archive = [head.parser for head in heads]
                num_matches = parser_names_in_archive.count(current_parser)
                if num_matches == 0:
                    raise ValueError(
                        f"No heads available for {current_parser} in the NN Archive."
                    )
                elif num_matches == 1:
                    head = [
                        head for head in heads if head.parser == current_parser
                    ][0]
                else:
                    raise ValueError(
                        f"Multiple heads with parser= {current_parser} detected, please specify a head name."
                    )

        parser_name = head.parser
        metadata = head.metadata
        outputs = head.outputs

        if outputs is None:
            raise ValueError(
                f"{head_name} head does not have any outputs specified."
            )

        head_dictionary = {}
        head_dictionary["parser"] = parser_name
        head_dictionary["outputs"] = outputs
        if metadata is not None:
            head_dictionary.update(metadata.extraParams)
        self.head_config = head_dictionary

        return self

    def run(self, **kwargs):
        """Parses the output from the neural network head.

        This method should be overridden by subclasses to implement the specific parsing logic.
        It accepts arbitrary keyword arguments for flexibility.

        Args:
            **kwargs: Arbitrary keyword arguments for the parsing process.

        Returns:
            The parsed output message, as defined by the logic in the subclass.
        """
        raise NotImplementedError(
            "Missing the parsing logic. Implement the run method."
        )
