import argparse
import logging
import coalFLUT
import pyFLUT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='runCoalFLUTeq',
        description=('Generate a multi-dimensional FLUT for'
                     ' coal combustion based on equilibrium')
    )
    parser.add_argument('yml', action='store',
                        default=None, help="YML input file")
    parser.add_argument('-d', dest='debug',
                        action='store_true',
                        help='Activate debug messages')
    parser.add_argument('-t', dest='turbulent', action='store_true',
                        help='Create table for turbulent flows')
    parser.add_argument('--hdf5', dest='hdf5',
                        action='store_true',
                        help=(
                            'Read the existing sldf.h5 file,'
                            ' previously created'))

    argument = parser.parse_args()

    # set the logging object
    loglevel = logging.DEBUG if argument.debug else logging.INFO

    logging.basicConfig(
        level=loglevel,
        format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")

    logging.info('runCoalFLUT use pyFLUT version: %s',
                 pyFLUT.__version__)

    # create FLUT object
    logging.debug('Initialize flut object')
    flut = coalFLUT.CoalFLUTEq(argument.yml)

    flut_file = flut.basename + '.h5'

    # generate flut
    if argument.hdf5:
        logging.info('Reading existing %s', flut_file)
        flut.read_h5(flut_file)
    else:
        logging.info('Generating FLUT...')
        flut.run_scoop()
    flut.squeeze()
    logging.debug('FLUT:\n%s', flut)

    # write bin
    logging.info('Save data to %s', flut_file)
    flut.write_bin(flut_file)

    # export to flameletConfig
    logging.info('Export data to FLUT.h5 (flameletConfig)')
    flut.write_hdf5(file_name='FLUT.h5',
                    turbulent=argument.turbulent, n_proc=1,
                    verbose=argument.debug)
