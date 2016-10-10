"""
coalPFLUT
========
"""
from coalFLUT import *
import multiprocessing as mp
import logging
from autologging import logged
import shutil
import pyFLUT.ulf as ulf
import pyFLUT
from termcolor import colored
import glob
import os
import yaml

from collections import *

def run_bs(fuel, oxidizer, parameters, par_format, ulf_reference, solver, species,
             key_names, basename='res', rerun=True):
    """
    Run a burner stabilized simulation with ULF

    Parameters
    ----------
    fuel: dict
        Fuel composition {T: 300, 'Y':{'CH4': 1}}
    oxidizer: dict
        Oxidizer composition {T: 300, 'Y':{'N2': 0.77, 'O2': 0.23}}
    parameters: dict
        Parameters of the simulations. {'Z':0.1, 'Y':0.5}.
        It must contains 'chist' as key
    par_format: str
        format string for saving chist in the result solutions
    ulf_reference: str
        reference file for the ulf simulation
    solver: str
        path of the ULF solver
    species: list
        list of species names
    key_names: dict
        dictionary containing keynames for ULF setup files:
        ('basename', 'chist')
    basename: str
        basename used for saving ulf results (default: 'res')
    rerun: bool
        Rerun if existing output file is found
    """
    logger = logging.getLogger('main.' + __name__ + '.run_bs')
    label = ''.join('_' + p + par_format.format(v)
                    for p, v in parameters.items())
    logger.debug('label %s -> %s', parameters, label)
    basename_run = basename + label
    logger.debug('basename_run: {}'.format(basename_run))
    basename_calc = basename_run + "_run"
    logger.debug('basename_calc: {}'.format(basename_calc))
    out_file = basename_run + '.ulf'
    logger.debug('out_file: %s', out_file)
    input_file = "inp" + label + ".ulf"
    logger.debug('input_file: {}'.format(input_file))
    shutil.copyfile(ulf_reference, input_file)
    runner = ulf.UlfRun(input_file, solver)
    runner[key_names['basename']] = basename_calc
    runner['Z_VALUE'] = parameters['Z']
    runner['TFIX'] = fuel['T']+20.0
    runner['SL_GUESS'] = fuel['u']
    pyFLUT.utilities.set_species(runner, fuel, oxidizer, species)
    if not rerun and os.path.exists(out_file):
        print(colored('Read existing file {}'.format(out_file), 'blue'))
        return pyFLUT.read_ulf(out_file)
    try:
        print('Running {}'.format(basename_calc))
        res = runner.run()
        logger.debug('End run')
        # set the run parameter -> it is necessary for aggregating the
        # data
        res.variables = parameters
        logger.debug('Parameters: {}'.format(res.variables))
        print(colored('End run {}'.format(basename_calc), 'green'))

        shutil.copyfile(basename_calc + 'final.ulf', out_file)
        logger.debug('Copy {} to {}'.format(
            basename_calc + 'final.ulf', out_file))
        files_to_delete = glob.iglob(basename_calc + '*')
        logger.debug('Delete: {}'.format(files_to_delete))
        [os.remove(f) for f in files_to_delete]
    except:
        print(colored('Error run {}'.format(basename_calc), 'red'))
        res = None

    return res


# New implementation #

@logged
class CoalPFLUT(CoalFLUT):
    streams = ['volatiles', 'oxidizer']

    def __init__(self, input_yml):
        super(CoalPFLUT, self).__init__(input_yml=input_yml)
        self.list_params = ['Hnorm', 'Y', 'Z']
        with open(input_yml, 'r') as f:
            self.__log.debug('Read YAML file %s', input_yml)
            input_dict = yaml.load(f)
        runner = ulf.UlfRun(
            file_input=self.ulf_reference, solver=self.solver)
        runner[self.keys['n_points']] = input_dict['ulf']['points']
        self.Z = pyFLUT.utilities.read_dict_list(
            **input_dict['flut']['Z'])
        self.VelRatio = pyFLUT.utilities.read_dict_list(
            **input_dict['flut']['VelRatio'])

    def assemble_data(self, results):
        '''
        Assemble flame1D ulf files to FLUT

        Parameters
        ----------
        results: list(ulf.UlfData)
           list of ulf.UlfData solutions with variables ['Hnorm', 'chist', 'Y']
        '''
        # create a new X grid -> non uniform X grid
        X_new = np.unique(
            np.sort(np.concatenate([r['X'] for r in results])))
        # ignore values < 0
        X_new = X_new[X_new>=0.0]
        self.__log.debug('Create X_new with %s points', len(X_new))
        numberOfResults=len(results)
        for idx in range(0,numberOfResults):
            r = results[idx]
            if r.data[0][0]<0.0:
                results[idx] = pyFLUT.Flame1D(output_dict=list(r.output_dict.keys()),
                           input_var=r.input_variables,
                           data=r.data[1:],
                           variables=r.variables)
                
        # interpolate existing solutions to the new grid
        [r.convert_to_new_grid(variable='X', new_grid=X_new)
         for r in results]
        super(pyFLUT.ulf.dflut.DFLUT_2stream, self).__init__(
            input_data=results, key_variable='X', verbose=True)
        self.__log.debug('Create data structure with dimension %s', self.ndim)
        for var, val in self.input_dict.items():
            self.__log.debug('%s defined from %s to %s with % points',
                             var, val[0], val[-1], len(val))
        idx  = self.input_variable_index('VelRatio')
        hMin = self['hMean'].min(axis=idx,keepdims=True)    
        hMax = self['hMean'].max(axis=idx,keepdims=True)    
        self['Hnorm'] = (self['hMean']-hMin)/(hMax-hMin)
        self=self.map_variables(from_inp='VelRatio',to_inp='Hnorm',verbose=True)

    def calc_progress_variable(self, definition_dict=None):
        '''
        Calculate progress variable. PV is defined in input_dict
        '''
        if not definition_dict:
            definition_dict = self.pv_definition
        super(pyFLUT.ulf.dflut.DFLUT_2stream, self).calc_progress_variable(
            definition_dict=definition_dict, along_variable='X')

    def run(self, n_p=1):
        """
        Run ulf for generating look-up tables running ulf solutions

        Parameters
        ----------
        n_p: int:
          number of processes
        """
        if n_p > 1:
            use_mp = True
            pool = mp.Pool(processes=n_p)
            res_async = []
        else:
            use_mp = False
            results = []

        oxid = {}
        oxid['Y'] = self.oxidizer['Y']
        for Y in self.Y:
            mix = self.mix_streams(Y)
            self.__log.debug(
                'Y=%s H_mix=%s', Y, mix['H'])

            for Z in self.Z:
                self.__log.debug('Y=%s Z=%s', Y, Z)
                fuel = mix.copy()
                fuel['T'] = fuel['T'][0]
                self.__log.debug('Tf=%s', fuel['T'])
                oxid = self.oxidizer.copy()
                oxid['T'] = oxid['T'][0]
                self.__log.debug('Toxidizer=%s', oxid['T'])
                #run freely propagating for Hnorm = 1
                VelRatio = 1.0
                fuel['u'] = 0.01
                parameters = OrderedDict()
                parameters['Y'] = Y
                parameters['Z'] = Z
                parameters['VelRatio'] = VelRatio
                args = (fuel,
                        oxid,
                        parameters,
                        self.format,
                        "fp_setup.ulf",
                        self.solver,
                        self.gas.species_names,
                        self.keys,
                        self.basename,
                        self.rerun)
                if use_mp:
                    res_async.append(
                        pool.apply_async(run_bs, args=args))
                else:
                    results.append(run_bs(*args))
                    sL = results[-1]['u'][0]
                    hMean = results[-1]['hMean'][-1]
                    print(colored('serial: sL is {}'.format(sL), 'magenta'))
                    for VelRatio in self.VelRatio[:-1]:
                        fuel['u'] = VelRatio*sL 
                        print(colored('VelRatio is {}. u is {}'.format(VelRatio,fuel['u']), 'yellow'))
                        parameters = OrderedDict()
                        parameters['Y'] = Y
                        parameters['Z'] = Z
                        parameters['VelRatio'] = VelRatio
                        args = (fuel,
                                oxid,
                                parameters,
                                self.format,
                                self.ulf_reference,
                                self.solver,
                                self.gas.species_names,
                                self.keys,
                                self.basename,
                                self.rerun)

                        results.append(run_bs(*args))

        if use_mp:
            numberfpRuns=len(res_async)
            for run in range(0,numberfpRuns):
                tmp = res_async[run].get();
                sL = tmp['u'][0]
                Y = tmp['Y']
                Z = tmp['Z']
                print(colored('parallel: sL is {}, Y is {}, Z is {}'.format(sL,Y,Z), 'magenta'))
                for VelRatio in self.VelRatio[:-1]:
                    fuel['u'] = VelRatio*sL 
                    print(colored('VelRatio is {}. u is {}'.format(VelRatio,fuel['u']), 'yellow'))
                    parameters = OrderedDict()
                    parameters['Y'] = Y
                    parameters['Z'] = Z
                    parameters['VelRatio'] = VelRatio
                    args = (fuel,
                            oxid,
                            parameters,
                            self.format,
                            self.ulf_reference,
                            self.solver,
                            self.gas.species_names,
                            self.keys,
                            self.basename,
                            self.rerun)
                    res_async.append(
                        pool.apply_async(run_bs, args=args))

            results = [r.get() for r in res_async]

        self.assemble_data(results)
