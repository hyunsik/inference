"""
furiosasdk backend (https://github.com/microsoft/onnxruntime)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from furiosa.runtime import session, __version__

import backend


class BackendFuriosa(backend.Backend):
    def __init__(self):
        super(BackendFuriosa, self).__init__()

    def version(self):
        return __version__

    def name(self):
        """Name of the runtime."""
        return "furiosa"

    def image_format(self):
        """image_format. For onnx it is always NCHW."""
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""

        self.sess = session.create(model_path)
        # get input and output names
        if not inputs:
            self.inputs = [meta.name for meta in self.sess.inputs()]
        else:
            self.inputs = inputs
        if not outputs:
            self.outputs = [meta.name for meta in self.sess.outputs()]
        else:
            self.outputs = outputs
        return self

    def predict(self, feed):
        """Run the prediction."""
        return self.sess.run(feed[None])
