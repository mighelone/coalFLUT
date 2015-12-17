import sys
sys.path.insert(0, '../')

import coalFLUT

if __name__ == "__main__":
    res = coalFLUT.coalFLUT('input.yml')
    res.run(n_p=4)
    res.write_bin('coalFLUT.h5')