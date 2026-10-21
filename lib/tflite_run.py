"""Runs one .tflite file with the LiteRT interpreter (pip install ai-edge-litert); falls back to TensorFlow's interpreter if LiteRT is not installed.
Models with integer (int8) input or output are handled with the model's own scale and zero point."""
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
    RUNTIME = "LiteRT"
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    RUNTIME = "TensorFlow Lite"


class Model:
    def __init__(self, path, threads=4, nc=80):
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.nc = nc

    def __call__(self, x_nchw):
        """x_nchw: float32 1 x 3 x S x S (0-1). Returns the raw detections (N, 5 + nc) as float32."""
        v = np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1), dtype=np.float32)
        d = self.inp
        if d["dtype"] != np.float32:
            scale, zero = d["quantization"]
            info = np.iinfo(d["dtype"])
            v = np.clip(np.round(v / scale) + zero, info.min, info.max).astype(d["dtype"])
        self.it.set_tensor(d["index"], v)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])
        if self.out["dtype"] != np.float32:
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y.reshape(-1, 5 + self.nc)
