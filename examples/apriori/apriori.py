from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import coal_calculation
import pyFLUT

c = coal_calculation.Coal1D('coal1D_alpha0.5000_U0.3000.ulf')

flut = pyFLUT.Flut.read_h5('coalFLUT-cc.h5')

# calc H_o H_f
H_o = [flut.extract_values('hMean', Z=0, Hnorm=h) for h in (0, 1)]
H_f = [flut.extract_values('hMean', Z=1, Hnorm=h) for h in (0, 1)]
c.calc_Hnorm(H_o, H_f)

c.plotdata('X', 'Hnorm')

# progress variable
c.calc_progress_variable(definition_dict={'CO': 1, 'CO2': 1})

# a priori temperature non corrected temp
T_nc = c.calc_a_priori_analysis('T', flut, correct=False)
T_c = c.calc_a_priori_analysis('T', flut, correct=True)

x = c['X']
ax = c.plotdata('X', 'T', label='FTC')
ax.plot(x, T_nc, marker='o', linewidth=0, markevery=20, label='NC')
ax.plot(x, T_c, marker='<', linewidth=0, markevery=20, label='C')

ax.legend()
