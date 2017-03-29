import pytest

from coalFLUT.premix import CoalPremixFLUT


settings = {'flut': {'c': 101,
                     'pv': {'CO': 1, 'CO2': 1},
                     'solver': 'OpenFOAM',
                     'varZ': {'method': 'linspace', 'values': [0, 1, 3]},
                     'variables': ['rho', 'T']},
            'oxidizer': {'T': {'max': 500, 'min': 300}, 'X': {'N2': 0.79, 'O2': 0.21}},
            'parameters': {'Hnorm': {'method': 'linspace', 'values': [0, 1, 2]},
                           'Y': {'method': 'linspace', 'values': [0, 1, 3]},
                           'chist': {'method': 'logspace', 'values': [-1, 1, 10]}},
            'pressure': 101325,
            'ulf': {'basename': 'test',
                    'keys': {'Zst': 'ZST',
                             'basename': 'RESULT_BASENAME',
                             'chist': 'CHIST',
                             'mechanism': 'MECHANISM',
                             'n_grid': 'REFINE_TARGET',
                             'pressure': 'PRESSURE'},
                    'tpatch_end': 500,
                    'mechanism': 'ch4_smooke.xml',
                    'n_grid': 101,
                    'par_format': '{:5.4f}',
                    'restart': True,
                    'solver': '/shared_home/vascella/Codes/ulf2/build/ulfrun',
                    'ulf_reference': 'flamelet_setup.ulf'},
            'volatiles': {'T': {'max': 600, 'min': 300},
                          'Y': {'CH4': 0.5, 'H2': 0.2, 'N2': 0}}}
