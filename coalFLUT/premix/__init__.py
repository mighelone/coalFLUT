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
    _parameter_names = ['vel_ratio', 'Z', 'Y']
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

    def run(self, n_p=1):

        # the first time calculate solution for vel_ratio=1
        parameters = self._parameter.copy()
        parameters['vel_ratio'] = np.array([1.0])

        self.__log.info(
            'Start calculating freely propagating (fp) flames')
        results_fp = self._run_parameters(n_p, parameters)

        # sL is the array ZxY of the laminar flame speeds
        sL = np.zeros_like((len(self.Z), len(self.Y)))
        Z = self.Z.tolist()
        Y = self.Y.tolist()
        assert len(sL.ravel()) == len(results_fp), (
            "Number of "
            "results_fp different from YxZ")
        for flame in results_fp:
            i_Z = Z.index(flame.variables['Z'])
            i_Y = Y.index(flame.variables['Z'])
            sL[i_Z, i_Y] = flame['u'][0]

        self.sL = sL

        self.__log.info(
            'Start calculating bs flames')
        parameters = self._parameter.copy()
        parameters['vel_ratio'] = parameters['vel_ratio'][:-1]
        results_bs = self._run_parameters(n_p, parameters)

        self.assemble_results(results_fp + results_bs)

    def _run_set_runner(self, basename_calc, input_file, parameters):
        # define the runner and set running parameters
        self.__log.debug('Set runner %s', parameters)
        if self.gas is None:
            gas = cantera.Solution(self.mechanism)
        else:
            gas = self.gas
        vel_ratio, Z, Y = parameters
        # define the streams for the given Hnorm and Y
        mix = self.mix_fuels(Y, 0, gas)
        oxid = self.calc_temperature(self.oxidizer, 0, gas)

        # get laminar flame speed

        if vel_ratio < 1:
            # use bs setup
            shutil.copy(self.ulf_reference_bs, input_file)
            sL = self.sL[
                self.Z.tolist().index(Z),
                self.Y.tolist().index(Y)] * vel_ratio
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
