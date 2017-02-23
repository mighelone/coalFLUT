import coalFLUT
import logging

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
input_yml = 'input.yml'

flut = coalFLUT.CoalFLUTnew(input_yml)

# flut.run(n_p=2)
flut.read_files()

print flut
