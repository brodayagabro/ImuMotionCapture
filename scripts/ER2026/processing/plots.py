"""Optional static scientific figures from the exported temporal metric table."""
import csv
from pathlib import Path


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    with (output / 'comparison_10s.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    ids = list(dict.fromkeys(r['recording_id'] for r in rows))
    if not ids:
        raise ValueError('No temporal metrics to plot')
    fig, axes = plt.subplots(len(ids), 2, figsize=(13, 3.1 * len(ids)), squeeze=False)
    colors = ('#0072B2', '#E69F00', '#009E73', '#CC79A7')
    for row, rid in enumerate(ids):
        for col, kind in enumerate(('rotation_deg', 'direction_deg')):
            ax = axes[row, col]
            selected = [r for r in rows if r['recording_id'] == rid and r['kind'] == kind]
            joints = list(dict.fromkeys(r['joint'] for r in selected))
            for color, joint in zip(colors, joints):
                values = [r for r in selected if r['joint'] == joint]
                ax.plot([(float(r['start_s']) + float(r['end_s'])) / 2 for r in values],
                        [float(r['rms']) for r in values], color=color, label=joint)
            ax.set(title=rid.split('_', 2)[2], xlabel='Time, s', ylabel='RMS error, deg', ylim=(0, 185))
            ax.grid(alpha=.25)
            if row == 0:
                ax.legend(fontsize=8, ncol=2)
    axes[0, 0].set_title('Segment orientation in spine frame\n' + axes[0, 0].get_title())
    axes[0, 1].set_title('Bone direction in spine frame (ignores axial twist)\n' + axes[0, 1].get_title())
    fig.suptitle('Custom vs Noitom: original time grid, RMS in disjoint 10 s bins', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / 'comparison.png', dpi=150)
    fig.savefig(output / 'comparison.svg')
    plt.close(fig)
