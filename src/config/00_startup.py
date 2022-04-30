###################################################
# this is a custom file created by Ruan Pretorius #
# to customise the jupyter notebook environment.  #
###################################################

MY_PREFERENCE = 'light' # 'light', 'dark'


### For dark theme plots
# from: https://medium.com/@rbmsingh/making-jupyter-dark-mode-great-5adaedd814db
# see also: https://github.com/dunovank/jupyter-themes
from jupyterthemes import jtplot # theme = {onedork | grade3 | oceans16 | chesterish | monokai | solarizedl | solarizedd}
if MY_PREFERENCE == 'dark':	
	jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) # change theme to match with Jupyter Theme
elif MY_PREFERENCE == 'light':
	jtplot.style(theme='grade3', context='notebook', ticks=True, grid=False) # change theme to match with Jupyter Theme

### For custom matplotlib plots
# see: https://matplotlib.org/3.1.1/users/dflt_style_changes.html
# Defaults:
#import matplotlib as mpl
#mpl.rcParams['figure.figsize'] = [8.0, 6.0]
#mpl.rcParams['figure.dpi'] = 80
#mpl.rcParams['savefig.dpi'] = 100
#mpl.rcParams['font.size'] = 12
#mpl.rcParams['legend.fontsize'] = 'large'
#mpl.rcParams['figure.titlesize'] = 'medium'
# Custom:
import matplotlib as mpl
if MY_PREFERENCE == 'light':
	mpl.pyplot.style.use('default')

### Figure sizes
mpl.rcParams['figure.figsize'] = [10.0, 4.0]
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['figure.autolayout'] = True # use matplotlib.pyplot.tight_layout() by default

### Font sizes
SMALL_SIZE = 14
MEDIUM_SIZE = 15 +1
BIGGER_SIZE = 16 +2
mpl.rcParams['font.size']=MEDIUM_SIZE          # controls default text sizes
mpl.rcParams['axes.titlesize']=BIGGER_SIZE     # fontsize of the axes title
mpl.rcParams['axes.labelsize']=MEDIUM_SIZE    # fontsize of the x and y labels
mpl.rcParams['xtick.labelsize']=SMALL_SIZE    # fontsize of the tick labels
mpl.rcParams['ytick.labelsize']=SMALL_SIZE    # fontsize of the tick labels
mpl.rcParams['legend.fontsize']=SMALL_SIZE #SMALL_SIZE    # legend fontsize
#mpl.rcParams['figure.titlesize']=BIGGER_SIZE  # fontsize of the figure title
mpl.rcParams['axes.titlepad']=10.0 +6

### Plot Fonts
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

### Axes Ticks
SMALL_TICK = 4
BIG_TICK = 6
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

mpl.rcParams['axes.spines.right'] =   False			# draw right spine (axes border)
mpl.rcParams['axes.spines.top'] = 	  False			# draw top spine (axes border)
mpl.rcParams['axes.spines.left'] = 	  True			# draw left spine (axes border)
mpl.rcParams['axes.spines.bottom'] =  True			# draw bottom spine (axes border)

mpl.rcParams['xtick.top'] =           False    # draw ticks on the top side
mpl.rcParams['xtick.bottom'] =        True    # draw ticks on the bottom side
mpl.rcParams['xtick.major.size'] =    BIG_TICK     # major tick size in points
mpl.rcParams['xtick.minor.size'] =    SMALL_TICK       # minor tick size in points
mpl.rcParams['xtick.direction'] =     'out'   # direction: {in, out, inout}
mpl.rcParams['xtick.major.top'] =     False    # draw x axis top major ticks
mpl.rcParams['xtick.major.bottom'] =  True    # draw x axis bottom major ticks
mpl.rcParams['xtick.minor.top'] =     False    # draw x axis top minor ticks
mpl.rcParams['xtick.minor.bottom'] =  True    # draw x axis bottom minor ticks

mpl.rcParams['ytick.left'] =          True    # draw ticks on the left side
mpl.rcParams['ytick.right'] =         False    # draw ticks on the right side
mpl.rcParams['ytick.major.size'] =    BIG_TICK     # major tick size in points
mpl.rcParams['ytick.minor.size'] =    SMALL_TICK       # minor tick size in points
mpl.rcParams['ytick.direction'] =     'out'   # direction: {in, out, inout}
mpl.rcParams['ytick.major.left'] =    True    # draw y axis left major ticks
mpl.rcParams['ytick.major.right'] =   False    # draw y axis right major ticks
mpl.rcParams['ytick.minor.left'] =    True    # draw y axis left minor ticks
mpl.rcParams['ytick.minor.right'] =   False    # draw y axis right minor ticks

### Legend
mpl.rcParams['legend.frameon'] =      False    # if True, draw the legend on a background patch


### Seaborn Plots
### from: https://stackoverflow.com/questions/25451294/best-way-to-display-seaborn-matplotlib-plots-with-a-dark-ipython-notebook-profil
if MY_PREFERENCE == 'dark':
	import seaborn as sns
	sns.set_style(None, rc=None)