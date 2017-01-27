#!/usr/bin/env python


import argparse
import logging
import coalFLUT
import pyFLUT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='runCoalFLUT',
        description=('Generate a multi-dimensional FLUT for'
                     ' coal combustion based on non-premixed flamelet')
    )
    parser.add_argument('yml', action='store',
                        default=None, help="YML input file")
    parser.add_argument('--hdf5', dest='hdf5',
                        action='store_true',
                        help=(
                            'Read the existing sldf.h5 file,'
                            ' previously created'))
    parser.add_argument('-d', dest='debug',
                        action='store_true',
                        help='Activate debug messages')
    parser.add_argument('--run-only', dest='run_only',
                        action='store_true',
                        help='Run only ULF and agrregate solutions')
    parser.add_argument('--readfiles', dest='readfiles',
                        action='store_true',
                        help='Read existing files')
    parser.add_argument('--fc', dest='flamelet_config', type=str,
                        action='store',
                        default=None, help='Export to flameletConfig')
    parser.add_argument('-t', dest='turbulent', action='store_true',
                        help='Create table for turbulent flows')
    parser.add_argument('--noquench', dest='noquench',
                        action='store_true',
                        help='Do not include quenched solutions')

    argument = parser.parse_args()

    flamelet_config = argument.flamelet_config

    loglevel = logging.DEBUG if argument.debug else logging.INFO

    logging.basicConfig(
        level=loglevel,
        format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")

    logging.info('runCoalFLUT use pyFLUT version: %s',
                 pyFLUT.__version__)
    logging.debug('flamelet_config %s', flamelet_config)
    logging.debug('Initialize flut object')

    flut = coalFLUT.CoalFLUT(argument.yml)

    flut_file = flut.basename + '.h5'
    if argument.readfiles:
        logging.debug('Read existing files')
        flut.read_files(argument.noquench)
    elif argument.hdf5:
        logging.debug('Read previous hdf5 %s', flut_file)
        flut.read_bin(flut_file)
    else:
        logging.debug('Run ULF')
        flut.run_scoop()
        logging.debug('Finished Run ULF')
    logging.debug('Results: \n%s', flut.__str__())
    flut.squeeze()
    logging.debug('Results after squeezing: \n%s', flut.__str__())

    # process table
    if not argument.run_only:
        along = 'X'

        # calc progress variable
        if 'cc' not in flut.input_dict:
            logging.debug('Calc progress variable %s',
                          flut.pv_definition)
            flut.calc_progress_variable()
            flut = flut.convert_cc_to_uniform_grid(n_points=len(flut.cc))
            along = 'Z'
        if all('Le_{}'.format(sp) in flut for sp in flut.pv_definition):
            logging.debug('Calc Le_yc')
            flut.calc_Le_yc(along=along)
            flut.export_variables += ['Le_yc']
        logging.debug('Add missing properties')
        flut.add_missing_properties(verbose=True)
        if not argument.hdf5:
            logging.debug('Write h5 file %s', flut_file)
            flut.write_bin(flut_file)
        else:
            logging.debug('Don\'t write h5 file')

        if flamelet_config:
            logging.debug('Export data to HDF5 flamelet_config %s',
                          flamelet_config)
            flut.write_hdf5(file_name=flamelet_config,
                            turbulent=argument.turbulent, n_proc=n_p)
