from coalFLUT.diffusion import CoalDiffusionFLUT
import logging

logging.basicConfig(level=logging.INFO)

flut = CoalDiffusionFLUT('test.yml')
logging.warning('parameters: %s', flut._parameter_names)

# flut._run_function((1.0, 0.1, 0.0))
flut.run(n_p=8)

logging.warning(flut)
