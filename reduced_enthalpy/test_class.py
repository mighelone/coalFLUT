import pyFLUT.ulf as ulf
from pyFLUT.ulf.ulfSeriesReader import ulf_to_cantera
import numpy as np
import sys
sys.path.insert(0, '../')
import coalFLUT
import cantera


res = coalFLUT.coalFLUT('input.yml')
res.run()
res.read_bin('coalFLUT.h5')

res.extend_enthalpy_range([-2])
res.write_bin('coalFLUT_ext.h5')