from manolo.base.wrappers import version_test
import optimum.quanto; version_test(optimum.quanto)

from optimum.quanto import quantize, Calibration, freeze, quantization_map
from optimum.quanto import qint2, qint4, qint8, qfloat8 
