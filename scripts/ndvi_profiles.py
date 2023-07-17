import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_parquet('../data/culture_dataset.parquet')
dataframe = dataframe.drop('TRAIN', axis=1)

dates = ['2020-12-22',
         '2020-12-27',
         '2021-01-03',
         '2021-01-18',
         '2021-01-26',
         '2021-02-15',
         '2021-03-14',
         '2021-03-22',
         '2021-03-24',
         '2021-04-13',
         '2021-04-18',
         '2021-05-06',
         '2021-05-18',
         '2021-05-21']

vis = [f'V{i+1}' for i in range(14)]
target_class = 'culture'
ndvi = dataframe[vis + ['filiere', 'culture']].query('filiere=="cereales"').copy()
plt.rcParams['font.family'] = 'Linux Libertine'
plt.rcParams['font.size'] = 14
fig, axs = plt.subplots(3, figsize=(10, 10), sharex='col')
labels = ndvi[f'{target_class}'].unique().tolist()
for label in labels:
    df = ndvi[ndvi[f'{target_class}'] == label].median(numeric_only=True)
    df.plot(label=label, linestyle='dashdot', ax=axs[0])
    x = [i for i in range(14)]
    y = df.tolist()
    axs[0].scatter(x, y, marker='x')
axs[0].set(ylabel='NDVI')
Line, Label = axs[0].get_legend_handles_labels()
Label = ['Avoine', 'Blé tendre', 'Blé dur', 'Orge']
fig.legend(Line, Label, loc='upper right',
           bbox_to_anchor=(1.09, 0.85),
           fontsize=14)


ndvi = dataframe[vis + ['filiere', 'culture']].query('filiere=="legumineux"').copy()
labels = ndvi[f'{target_class}'].unique().tolist()
for label in labels:
    df = ndvi[ndvi[f'{target_class}'] == label].median(numeric_only=True)
    df.plot(label=label, linestyle='dashdot', ax=axs[1])
    x = [i for i in range(14)]
    y = df.tolist()
    axs[1].scatter(x, y, marker='x')
axs[1].set(ylabel='NDVI')
Line, Label = axs[1].get_legend_handles_labels()
Label = ['Fèverole', 'Pois chiche']
fig.legend(Line, Label, loc='upper right',
           bbox_to_anchor=(1.1, 0.53),
           fontsize=14)

ndvi = dataframe[vis + ['filiere', 'culture']].query('filiere=="arboriculture"').copy()
labels = ndvi[f'{target_class}'].unique().tolist()
for label in labels:
    df = ndvi[ndvi[f'{target_class}'] == label].median(numeric_only=True)
    df.plot(label=label, linestyle='dashdot', ax=axs[2])
    x = [i for i in range(14)]
    y = df.tolist()
    axs[2].scatter(x, y, marker='x')
axs[2].set(ylabel='NDVI')
Line, Label = axs[2].get_legend_handles_labels()
Label = ['Agrumes', 'Grenadier']
fig.legend(Line, Label, loc='upper right',
           bbox_to_anchor=(1.09, 0.27),
           fontsize=14)


axs[0].xaxis.set_ticks_position('none')
axs[1].xaxis.set_ticks_position('none')
axs[2].set_xticks([i for i in range(14)])
axs[2].set_xticklabels(dates, rotation=45)

plt.subplots_adjust(wspace=0, hspace=0.05)


def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[1] * scale, ax.get_ylim()[1] * scale


axs[0].annotate('(a)', xy=get_axis_limits(axs[0]))
axs[1].annotate('(b)', xy=get_axis_limits(axs[1]))
axs[2].annotate('(c)', xy=get_axis_limits(axs[2]))

plt.savefig('../fig/profils/ndvi_composite.png', dpi=300, bbox_inches='tight')
