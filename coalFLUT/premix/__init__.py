"""
Coal Premix FLUT
"""
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from ..abstract import AbstractCoalFLUT
from autologging import logged

import numpy as np
import cantera
import pyFLUT
import pyFLUT.ulf
import shutil


@logged
class CoalPremixFLUT(AbstractCoalFLUT):
    """
    Coal FLUT using premix flames
    """

    _along = 'X'
    _key_variable = 'X'
    _parameter_names = ['velratio', 'Z', 'Y']
    _key_names_extra = ('tpatch_end', 'Z', 'T_fix', 'sl_guess')
    _ulf_parameters_extra = ('tpatch_end', 'ulf_reference_bs', 'Le1',
                             'T_fix', 'sl_guess')
    _files = ('bs_setup.ulf',
              'bs_template.ulf',
              'fp_mixtureEntries.ulf',
              'fp_new_template.ulf',
              'fp_setup.ulf',
              'hReactor_relax.ulf',
              'ch4_smooke.xml',
              'input.yml')

    def set_runner(self):
        super(CoalPremixFLUT, self).set_runner()

        # temporary change the ulf_reference
        ulf_reference = self.ulf_reference
        self.ulf_reference = self.ulf_reference_bs

        # set ulf_reference_bs
        super(CoalPremixFLUT, self).set_runner()
        self.ulf_reference = ulf_reference

    def run(self, n_p=1):

        # the first time calculate solution for velratio=1
        parameters = self._parameter.copy()
        parameters['velratio'] = np.array([1.0])

        self.__log.info(
            'Start calculating freely propagating (fp) flames')
        results_fp = self._run_parameters(n_p, parameters)

        # sL is the array ZxY of the laminar flame speeds
        sL = np.zeros((len(self.Z), len(self.Y)))
        Z = self.Z.tolist()
        Y = self.Y.tolist()
        assert len(sL.ravel()) == len(results_fp), (
            "Number of "
            "results_fp different from YxZ")
        for flame in results_fp:
            i_Z = Z.index(flame.variables['Z'])
            i_Y = Y.index(flame.variables['Y'])
            sL[i_Z, i_Y] = flame['u'][0]

        self.sL = sL

        self.__log.info(
            'Start calculating bs flames')
        parameters = self._parameter.copy()
        parameters['velratio'] = parameters['velratio'][:-1]
        results_bs = self._run_parameters(n_p, parameters)
        # results_bs = [self._cut_flame(res) for res in results_bs]

        self.assemble_results(results_fp + results_bs)

    def _run_set_runner(self, basename_calc, input_file, parameters):
        # define the runner and set running parameters
        self.__log.debug('Set runner %s', parameters)
        if self.gas is None:
            gas = cantera.Solution(self.mechanism)
        else:
            gas = self.gas
        velratio, Z, Y = parameters
        # define the streams for the given Hnorm and Y
        mix = self.mix_fuels(Y, 0, gas)
        oxid = self.calc_temperature(self.oxidizer, 0, gas)

        # get laminar flame speed

        if velratio < 1:
            # use bs setup
            shutil.copy(self.ulf_reference_bs, input_file)
            sL = self.sL[
                self.Z.tolist().index(Z),
                self.Y.tolist().index(Y)] * velratio
        else:
            sL = self.sl_guess

        runner = pyFLUT.ulf.UlfRun(input_file, self.solver)
        runner.set(self.keys['basename'], basename_calc)
        runner.set(self.keys['sl_guess'], sL)
        runner.set(self.keys['Z'], Z)
        runner.set(self.keys['T_fix'], mix['T'] + self.T_fix)

        self.__log.debug('Set species mix: %s', mix)
        pyFLUT.utilities.set_species(runner, mix, oxid,
                                     gas.species_names)

        return runner

    def assemble_results(self, results, verbose=True):
        for i, res in enumerate(results):
            if res.variables['velratio'] < 1:
                # remove points at const T
                # shift and extend the grid
                # in bs solutions
                results[i] = self._cut_flame(res)
            else:
                # add deltah to fp solutions
                res['deltah'] = 0
        super(CoalPremixFLUT, self).assemble_results(results, verbose)

    def convert_cc_to_uniform_grid(self, output_variables=None, n_points=None,
                                   n_proc=1, verbose=False):
        if verbose:
            self.__log.info('Map {} to cc'.format(self.along))
        flut_cc = super(CoalPremixFLUT, self).convert_cc_to_uniform_grid(
            output_variables, n_points, n_proc, verbose)

        index = self.input_variable_index('velratio')
        # hMean_min
        flut_cc['hMean_min'] = 0
        flut_cc['hMean_min'] = np.min(flut_cc['hMean'], axis=index,
                                      keepdims=True)
        # hMean_max
        flut_cc['hMean_max'] = 0
        flut_cc['hMean_max'] = np.max(flut_cc['hMean'], axis=index,
                                      keepdims=True)
        # Hnorm
        flut_cc['Hnorm'] = ((flut_cc['hMean'] - flut_cc['hMean_min']) /
                            (flut_cc['hMean_max'] - flut_cc['hMean_min']))
        # map from velratio to Hnorm
        if verbose:
            self.__log.info('Map velratio to Hnorm')
        return flut_cc.map_variables(
            from_inp='velratio', to_inp='Hnorm', n_points=11, verbose=verbose,
            n_proc=n_proc)
        # return flut_cc

    @staticmethod
    def _cut_flame(flame):
        last_X = np.where(flame['T'] == flame['T'][0])[0][-1]
        data = flame.data[slice(last_X, -1), :]
        cutted_flame = pyFLUT.Flame1D(data=data,
                                      input_var='X',
                                      output_dict=flame.output_dict,
                                      variables=flame.variables)
        cutted_flame['X'] = cutted_flame['X'] - cutted_flame['X'][0]
        return cutted_flame.extend_length(flame['X'][-1])

    def _exception_ulf_fail(self, parameters, basename_calc):
        """
        Manage the exception when running ulf. This create a quenched solution
        starting from the init ulf file, extending the inlet conditions.
        The flame created has dimension 2 x n_out.
        Solution is also dumped in a file.

        Parameters
        ----------
        parameters: tuple
            Simulation parameters
        basename_calc: str
            Basename used for the calculation

        Returns
        -------
        pyFLUT.Flame1D
        """
        res_init = pyFLUT.Flame1D.read_ulf(basename_calc + 'init.ulf')
        data = np.empty((2, len(res_init.output_variables)))
        data[0] = res_init.data[0]
        data[-1] = res_init.data[0]
        data[-1, res_init.output_dict['X']] = res_init['X'][-1]
        res = pyFLUT.Flame1D(
            data=data, output_dict=res_init.output_variables,
            variables={par: value for par, value in
                       zip(self._parameter_names, parameters)})
        res['deltah'] = 0
        # dump the file
        res.write_ascii(self.basename + self.create_label(parameters) + '.ulf')
        return res
