"""Cubic BN: phonons, branch-resolved phonon-assisted PL, and convergence in the intermediate excitons (main text).

(a) LDA phonon dispersion L-G-X-W-K (q2r/matdyn from the 16^3 production DFPT, no Born charges: LO-TO splitting
    at Gamma absent, irrelevant at X); dots: the four phonons at X that make the replicas.
    data/cBN/phonon/cbn.freq.gp (path L(87) G(100) X(50) W(35) K(1) points)
(b) PL at 10 K, 16^3 grid, 600 intermediate Q=0 excitons, 12 emitting per Q, 5 meV, resolved by branch.
    data/cBN/cBN_PL_branches600_16k.npz (keys {TA,LA,TO,LO,total}_n600_T10_b5)
(c) Replica heights relative to TO versus the number n_in of intermediate states (12-600); dotted line: n_in = 213,
    where the bright triplet at 11.39 eV (states 211-213) enters.
    data/cBN/cBN_PL_nexc_conv_16k.npz, _nexc400_, _nexc600_ (keys PL_n{n}_T10); heights = local max within 2 meV of
    E_X - omega (E_X = 6.031 eV).
Output: Figures/paper/fig_cbn.pdf/.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from paperstyle import COL2, DATA, OUT, BLUE, ORANGE, PURPLE, CYAN, RED, panel

BCOL = {'TA': BLUE, 'TO': ORANGE, 'LA': PURPLE, 'LO': CYAN}
CM2MEV = 0.12398419843320026
EX = 6.031

fig, ax = plt.subplots(1, 3, figsize=(COL2, 2.35), gridspec_kw=dict(width_ratios=[1.15, 1.1, 1.0], wspace=0.38))

# (a) phonons
gp = np.loadtxt(os.path.join(DATA, 'phonon', 'cbn.freq.gp'))
x, w = gp[:, 0], gp[:, 1:] * CM2MEV
iL, iG, iX, iW, iK = 0, 87, 187, 237, len(x) - 1
for b in range(w.shape[1]):
    ax[0].plot(x, w[:, b], color='k', lw=1.1)
for i in (iG, iX, iW):
    ax[0].axvline(x[i], color='0.75', lw=0.6)
wx = np.sort(w[iX])
OM = {'TA': wx[0], 'TO': wx[2], 'LA': wx[4], 'LO': wx[5]}
for lab, val in OM.items():
    ax[0].plot(x[iX], val, 'o', color=RED, ms=3.5, zorder=5)
    ax[0].annotate('%s %.0f meV' % (lab, val), (x[iX], val), xytext=(4, -6 if lab in ('TO',) else (2 if lab == 'LO' else -3)),
                   textcoords='offset points', fontsize=6.5, color=BCOL[lab], bbox=dict(fc='white', ec='none', pad=0.2))
ax[0].set_xticks([x[i] for i in (iL, iG, iX, iW, iK)]); ax[0].set_xticklabels(['L', r'$\Gamma$', 'X', 'W', 'K'])
ax[0].set_xlim(x[0], x[-1]); ax[0].set_ylim(0, 165); ax[0].set_ylabel('Phonon energy (meV)')
panel(ax[0], 'a')

# (b) branch-resolved spectrum
cb = np.load(os.path.join(DATA, 'cBN_PL_branches600_16k.npz'))
wp = cb['omega']; tot = cb['total_n600_T10_b5']; s = tot.max()
for b in ('TA', 'TO', 'LA', 'LO'):
    y = cb['%s_n600_T10_b5' % b] / s
    ax[1].fill_between(wp, 0, y, color=BCOL[b], alpha=0.3, lw=0)
    ax[1].plot(wp, y, color=BCOL[b], lw=1.0, label=b)
ax[1].plot(wp, tot / s, color='k', lw=1.1, label='total')
ax[1].axvline(EX, color='k', ls='--', lw=0.8)
ax[1].set_xlim(5.865, 6.04); ax[1].set_ylim(0, 1.08)
ax[1].set_xlabel('Energy (eV)'); ax[1].set_ylabel('PL (norm.)')
ax[1].legend(frameon=False, fontsize=6.5, loc='upper center', handlelength=1.0, ncol=1)
panel(ax[1], 'b', '10 K')

# (c) convergence in n_in
spec = {}
for f in ('cBN_PL_nexc_conv_16k.npz', 'cBN_PL_nexc400_conv_16k.npz', 'cBN_PL_nexc600_conv_16k.npz'):
    d = np.load(os.path.join(DATA, f))
    for n in d['nlist']:
        spec[int(n)] = (d['omega'], d['PL_n%d_T10' % n])
ns = sorted(spec)


def height(wv, p, e):
    m = np.abs(wv - e) < 0.002
    return p[m].max()


for b in ('TA', 'TO', 'LA', 'LO'):
    H = [height(*spec[n], EX - OM[b] / 1e3) / height(*spec[n], EX - OM['TO'] / 1e3) for n in ns]
    ax[2].plot(ns, H, 'o-', color=BCOL[b], ms=3.0, lw=1.0, label=b)
ax[2].axvline(213, color='0.5', ls=':', lw=0.9)
ax[2].text(225, 0.45, 'bright\ntriplet\n11.4 eV', fontsize=6.5, color='0.35', va='center')
ax[2].set_xscale('log'); ax[2].set_ylim(0, 1.1)
ax[2].set_xlabel(r'intermediate states $n_{\rm in}$'); ax[2].set_ylabel('replica / TO')
panel(ax[2], 'c')

fig.savefig(os.path.join(OUT, 'fig_cbn.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_cbn.png'), bbox_inches='tight', dpi=200)
print('phonons at X (meV):', {k: round(float(v), 1) for k, v in OM.items()}, ' n_in:', ns)
