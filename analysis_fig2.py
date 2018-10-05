import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# The following two lines ensure type 1 fonts are used in saved pdfs
mpl.rcParams['pdf.fonttype'] = 42
# plotting defaults
plt_params = {'font.size'   : 14}
plt.rcParams.update(plt_params)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


results1 = np.load('results/shifted_centers-sigmas_00-90deg_40-60cm.npy').item()
results2 = np.load('results/shifted_centers-sigmas_00-90deg_40-60cm_short-hard-kick.npy').item()


row_keys = ['centers', 'sigmas']
col_keys = ['40cm', '60cm']
col_keys_exp = ['I1', 'I2']
line_keys  = ['00deg', '90deg']
line_keys_exp = ['00', '90']

line_labels = [None, r'$\mathrm{coils}~@~90^\circ$',  r'$\mathrm{coils}~@~0^\circ$']

def plot_exp(ax, Inum, deg, measure, hscale=1600.0/800.0, sub_nokick=False):
    file_template = 'experiment/kick_summary/{}{}_{}.csv'
    keys = ['Vs', 'centers', 'dcenters', 'sigmas', 'dsigmas'] # csv header
    reader = csv.reader(open(file_template.format('kick_', Inum, deg), 'r'))
    nokick_reader = csv.reader(open(file_template.format('no_kick_', Inum, deg), 'r'))
    next(reader) # skip headers
    next(nokick_reader)
    nokick_row = ['0.0'] + list(next(nokick_reader))
    nkd = {k : eval(nkv) for k, nkv in zip(keys, nokick_row)}
    d = {k : [] for k in keys}
    for row in reader:
        for k, v in zip(keys, row):
            d[k] += [eval(v)]
    x = hscale * np.array(d['Vs'])
    y = np.array(d[measure])
    dy = np.array(d['d'+measure])
    y0 = nkd[measure]
    dy0 = nkd['d' + measure]
    if sub_nokick:
        if measure == 'sigmas':
            dy = np.sqrt( (y**2 * dy**2 + y0**2 * dy0**2) / (y**2 - y0**2) )
            y = np.sqrt(y**2 - y0**2)
            #y = y - y0
    if not sub_nokick:
        if measure == 'sigmas':
            ax.axhline(y0, c='k')
    ax.errorbar(x, y, yerr=dy,
            markersize=2, linestyle='', marker='^', c='k')


for sub_nokick in [True, False]:
    fig, axarr = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(7,4))
    ps = []
    for ii, (ls, version, results) in enumerate(zip(
            ['-', '--'], ['', '1D'], [results1])):
        xs = np.array(results['Vs']) / (10**ii)
        for row, row_key in enumerate(row_keys):
            ylabel = row_key[:-1] + ' [mm]'
            xlabel = 'bias voltage' + ' [V]'
            for col, (col_key, col_key_exp) in enumerate(zip(col_keys, col_keys_exp)):
                ax = axarr[row, col]
                for c, line_key, line_key_exp in zip(['c', 'm'], line_keys, line_keys_exp):
                    key = '_'.join([row_key, line_key, col_key])
                    ys = np.array(results[key])
                    y0 = results[key+'_nokick']
                    if sub_nokick:
                        if row_key == 'sigmas':
                            ys = np.sqrt(ys**2 - y0**2)
                            #ys = ys - y0
                    if not sub_nokick:
                        if row_key == 'sigmas':
                            ax.axhline(y0, c=c)
                    if (row, col) == (0, 0):
                        p = ax.plot(xs, ys, label=line_key+version, c=c, ls=ls)
                        ps += p
                    else:
                        ax.plot(xs, ys, c=c, ls=ls)
                    plot_exp(ax, col_key_exp, line_key_exp, row_key,
                            hscale=1600.0/800.0, sub_nokick=sub_nokick)
                    ax.set_ylabel(ylabel)
                    ax.set_xlabel(xlabel)
                    ax.grid(True)
                    plt.locator_params(axis='x', nbins=6)
                    ax.locator_params(axis='y', nbins=6)

    for ax in axarr.flat:
        ax.label_outer()
    axarr[0,0].set_ylim(-5, 5)
    #axarr[1,0].set_ylim(0.0, 1.0)
    axarr[0,0].set_title('40 cm from coils')
    axarr[0,1].set_title('60 cm from coils')
    axarr[0,0].legend(handles=ps, loc="lower left", handlelength=1, bbox_to_anchor=[2.2,0])
    axarr[0,0].text(0.85, 0.85, '(a)', transform = axarr[0,0].transAxes)
    axarr[0,1].text(0.85, 0.85, '(b)', transform = axarr[0,1].transAxes)
    axarr[1,0].text(0.85, 0.85, '(c)', transform = axarr[1,0].transAxes)
    axarr[1,1].text(0.85, 0.85, '(d)', transform = axarr[1,1].transAxes)

    plot_fname = 'plots/analysis/shifted_kick_compare_to_sim.pdf'
    if sub_nokick:
        plot_fname = 'plots/analysis/sub-nokick_shifted_kick_compare_to_sim.pdf'
    plt.savefig(plot_fname, bbox_inches='tight', extra_artists=True)
    print(plot_fname)

