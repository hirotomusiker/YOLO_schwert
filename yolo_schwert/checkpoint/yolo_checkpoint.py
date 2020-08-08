from fvcore.common.checkpoint import Checkpointer
from .parse_yolo_weights import parse_yolo_weights

class YOLOCheckpointer(Checkpointer):
    def _load_file(self, filename: str):
        if "darknet" in filename:
            # This parses the darknet weights on the backbone
            # and initialize the rest
            return parse_yolo_weights(self.model, filename, initflag=True)
        elif "yolov3" in filename:
            # This parses the yolov3 weights on the whole model
            return parse_yolo_weights(self.model, filename)

        # load the checkpoint normally
        loaded = super()._load_file(filename)
        print(filename, 'has been successfully loaded')
        return loaded