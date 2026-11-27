"""Shared style for the paper figures: matplotlib style, PRB widths, fixed colour-blind-safe colours.
Single paper palette (blue, orange, purple, cyan, grey). Import from the figN_*.py scripts."""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scienceplots  # noqa: F401  (registers the 'science' styles)

plt.style.use(['science', 'nature'])
mpl.rcParams.update({'savefig.dpi': 600, 'figure.dpi': 150, 'axes.linewidth': 0.7,
                     'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'lines.linewidth': 1.4,
                     'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
                     'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
                     'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
                     'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5})
FS_PANEL = 9      # panel labels (bold)
FS_ANNOT = 7.5    # in-panel annotations (irrep labels, state names)
CB_WIDTH = 0.012  # colorbar width, fraction of figure width

COL1 = 3.4   # PRB single column (in)
COL2 = 7.0   # PRB double column (in)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'figures')

# ONE palette for the whole paper, used in this order: blue, orange, purple, cyan, grey (+ dark grey).
BLUE = '#0072B2'
ORANGE = '#E69F00'
PURPLE = '#7B3294'
CYAN = '#56B4E9'
GREY = '#7F7F7F'
DGREY = '#3F3F3F'
PALETTE = [BLUE, ORANGE, PURPLE, CYAN, GREY, DGREY]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=PALETTE)

RED = '#B2182B'   # red / blue ends of DIVERGING (RdBu_r), for signed/two-group colour codes
DBLUE = '#2166AC'

# semantic aliases
SIGMA_PLUS = RED      # sigma+ <-> l_z > 0 (red end of DIVERGING)
SIGMA_MINUS = DBLUE   # sigma- <-> l_z < 0 (blue end)
DARK = GREY               # dark (dipole-forbidden) states
MAT = {'Diamond': BLUE, 'hBN-3D': ORANGE, 'cBN': PURPLE, 'hBN-2D': CYAN, 'bBN': GREY, 'MoTe2': DGREY,
       'MoS2': BLUE}
MO_D = RED                # orbital character colours (MoS2): Mo-d red, S-p blue
S_P = DBLUE

# paper-wide diverging map for signed quantities (l_z, <S_z>): blue (-) - white (0) - red (+), symmetric limits
DIVERGING = plt.get_cmap('RdBu_r')
# temperature ramp (no blue): 10 K dark plum -> red -> 300 K orange
_TRAMP = LinearSegmentedColormap.from_list('warm_ramp', ['#3B0F4F', RED, ORANGE])


def tcolor(T, Tmax=300.0):
    """Temperature -> colour: 10 K dark plum, 300 K orange (warm ramp, no blue)."""
    return _TRAMP(min(T, Tmax) / Tmax)


def panel(ax, letter, text=None, dx=0.0, dy=0.05):
    """Bold panel label (a) just above the top-left corner, outside the axes (offsets in inches, identical in every figure);
    optional plain description right after it."""
    from matplotlib.transforms import ScaledTranslation
    tr = ax.transAxes + ScaledTranslation(dx, dy, ax.figure.dpi_scale_trans)
    lab = r'\textbf{(%s)}' % letter if mpl.rcParams['text.usetex'] else '(%s)' % letter
    t = ax.text(0, 1, lab, transform=tr, ha='left', va='bottom', fontsize=FS_PANEL,
                weight='bold')
    if text:
        ax.text(0, 1, text, transform=ax.transAxes + ScaledTranslation(dx + 0.3, dy, ax.figure.dpi_scale_trans),
                ha='left', va='bottom', fontsize=FS_PANEL)
    return t


def colorbar(fig, ax, mappable, label=None, ticks=None, pad=0.01):
    """Colorbar with the common width CB_WIDTH (figure fraction), right of `ax`, same height."""
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + pad, pos.y0, CB_WIDTH, pos.height])
    cb = fig.colorbar(mappable, cax=cax, ticks=ticks)
    cb.outline.set_linewidth(0.7)
    if label:
        cb.set_label(label)
    return cb


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
    print('wrote', os.path.join(OUT, name + '.pdf'))
