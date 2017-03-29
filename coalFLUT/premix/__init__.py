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
        for res in results_fp:
            res['deltah'] = 0

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
