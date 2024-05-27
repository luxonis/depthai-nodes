# import nodes as needed, e.g.
#from .postprocess_node1 import PostProcessNode1
#from .postprocess_node2 import PostProcessNode2
from .postprocessing_node_zero_dce import ZeroDCEParser
from .postprocessing_node_dncnn3 import DnCNN3Parser
from .postprocessing_node_depth_anything import DepthAnythingParser
from .postprocessing_node_yunet import YuNetParser

#__all__ = ['PostProcessNode1', 'PostProcessNode2']