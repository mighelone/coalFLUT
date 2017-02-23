import coalFLUT
import pyFLUT
import mock
import numpy as np


def rd(fuel, oxidizer, parameters, par_format, ulf_reference, solver,
       species, key_names, basename='res'):
    output_dict = ['Z', 'T', 'rho', 'CO', 'CO2']
    ngrid = 101
    data = np.zeros((ngrid, len(output_dict)))
    data[:, 0] = np.linspace(0, 1, ngrid)
    return pyFLUT.Flame1D(output_dict=output_dict, data=data,
                          variables=parameters)


@mock.patch('coalFLUT.pyFLUT.ulf.dflut.run_sldf', side_effect=rd)
def test_mock(mocked_run_sldf):
    p = {'Y': 0.1, 'Hnorm': 0.5, 'chist': 0.1}
    res = coalFLUT.pyFLUT.ulf.dflut.run_sldf(fuel=None,
                                             oxidizer=None,
                                             parameters=p,
                                             par_format=None,
                                             ulf_reference=None,
                                             solver=None,
                                             species=None,
                                             key_names=None,
                                             basename=None)
    assert res.variables == p
    assert np.all(res['T'] == 0)
