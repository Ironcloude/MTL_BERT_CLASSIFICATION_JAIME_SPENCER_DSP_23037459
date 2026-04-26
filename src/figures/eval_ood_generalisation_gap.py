from pathlib import Path 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from theme import Colours, Fonts, DISPLAY_NAMES

style_path = Path(__file__).parent / 'style.mplstyle'
plt.style.use(style_path)


id_data= {
        "EX-5-DeB-st-lr-5e-05-100pct-512-20260405-164415": 0.87257,
        "EX-13-Mod-mt-λ0-lr-5e-05-100pct-1024-20260405-144629": 0.27258,
        "EX-12-Mod-mt-λ0.25-lr-5e-05-100pct-1024-20260405-125457": 0.85986,
        "EX-10-Mod-mt-λ0.75-lr-5e-05-100pct-1024-20260405-110558": 0.86407,
        "EX-9-Mod-mt-λ1.0-lr-5e-05-100pct-1024-20260405-090322": 0.87010,
        "EX-8-Mod-st-lr-5e-05-100pct-1024-20260405-064226": 0.57657,
        "EX-7-Mod-st-lr-5e-05-100pct-2048-20260405-013654": 0.87794,
        "EX-3-Mod-st-lr-5e-05-100pct-512-20260405-004029": 0.84823,
        "EX-4-ELE-st-lr-3e-05-100pct-512-20260404-235313": 0.84274,
        "EX-1-BER-st-lr-5e-05-100pct-512-20260404-231919": 0.83498,
        "EX-11-Mod-mt-λ0.5-lr-5e-05-100pct-1024-20260404-212731": 0.85221,
        "EX-6-Mod-st-lr-5e-05-100pct-1024-20260404-165535": 0.88344,
        "EX-2-DeB-st-lr-2.5e-05-100pct-512-20260404-143836": 0.85554
    }

eval_data = {
    "EX-1-BER-st-lr-5e-05-100pct-512-20260406-124204": 0.42756,
    "EX-2-DeB-st-lr-2.5e-05-100pct-512-20260407-170059": 0.423,
    "EX-3-Mod-st-lr-5e-05-100pct-512-20260406-133306": 0.45613,
    "EX-4-ELE-st-lr-3e-05-100pct-512-20260407-184202": 0.45363,
    "EX-5-DeB-st-lr-5e-05-100pct-512-20260406-142049": 0.44399,
    "EX-6-Mod-st-lr-5e-05-100pct-1024-20260406-165141": 0.4903,
    "EX-7-Mod-st-lr-5e-05-100pct-2048-20260406-181356": 0.49491,
    "EX-8-Mod-st-lr-5e-05-100pct-1024-20260406-211422": 0.41842,
    "EX-9-Mod-mt-λ1.0-lr-5e-05-100pct-1024-20260406-223702": 0.47616,
    "EX-10-Mod-mt-λ0.75-lr-5e-05-100pct-1024-20260406-232605": 0.46742,
    "EX-11-Mod-mt-λ0.5-lr-5e-05-100pct-1024-20260407-002522": 0.44929,
    "EX-12-Mod-mt-λ0.25-lr-5e-05-100pct-1024-20260407-012532": 0.4706,
    "EX-13-Mod-mt-λ0-lr-5e-05-100pct-1024-20260407-024836": 0.32684,
    "EX-14-Mod-mt-λ0.25-lr-5e-05-100pct-2048-20260407-125024": 0.44092,
}
def generate_dumbbell_plot(data, ood_csv_path: str, output_name, checkpoint_filter=None, labels=('In-Distribution', 'Out-of-Distribution'), 
                           title='ID Evaluation vs. Golden Set Performance',
                           xlim=(0, 1.1), xtick_step=0.1, ylabel_x=-0.02, use_axis_transform=False):
    df_eval = pd.DataFrame(list(data.items()), columns=['model_raw', 'EVAL_F1'])

    try:
        df_ood = pd.read_csv(ood_csv_path)
        if checkpoint_filter:
            df_ood = df_ood[df_ood['checkpoint'] == checkpoint_filter].copy()
        df_ood = df_ood[['model', 'macro_f1']].rename(
            columns={'model': 'model_raw', 'macro_f1': 'GS_F1'})
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Data Processing
    df = pd.merge(df_eval, df_ood, on='model_raw', how='inner')
    df['ex_id'] = df['model_raw'].apply(lambda x: "-".join(x.split("-")[:2]))
    df['Label_Tuple'] = df['ex_id'].map(DISPLAY_NAMES).fillna(df['ex_id'])
    df['Label_Tuple'] = df.apply(lambda row: (row['ex_id'], row['Label_Tuple']), axis=1)
    df = df.sort_values(by='GS_F1', ascending=True).reset_index(drop=True)

    # Plot Generation
    fig, ax = plt.subplots()

    ax.hlines(y=df.index, xmin=df['GS_F1'], xmax=df['EVAL_F1'],
              color=Colours.GREY_LIGHT, alpha=0.4, linewidth=2.5, zorder=1)

    ax.scatter(df['EVAL_F1'], df.index, color=Colours.BLUE, s=80, label=labels[0], zorder=2)
    ax.scatter(df['GS_F1'], df.index, color=Colours.RED, s=80, label=labels[1], zorder=3)

    for i, row in df.iterrows():
        delta = row['GS_F1'] - row['EVAL_F1']
        mid_point = (row['EVAL_F1'] + row['GS_F1']) / 2
        ax.text(mid_point, i + 0.15, f"{delta:+.3f}", **Fonts.DELTA)

    ax.set_yticks(df.index)
    ax.set_yticklabels([])
    text_kwargs = dict(transform=ax.get_yaxis_transform()) if use_axis_transform else {}
    for i, row in df.iterrows():
        ex_num, desc = row['Label_Tuple']
        ax.text(ylabel_x, i + 0.15, ex_num, **Fonts.EX_MAIN, **text_kwargs)
        ax.text(ylabel_x, i - 0.2, desc, **Fonts.EX_SUB, **text_kwargs)

    ax.set_xlabel('Macro F1 Score')
    fig.suptitle(title, x=0.5, ha='center')
    ax.set_xlim(*xlim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.01, xtick_step))

    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2, frameon=False)
    plt.subplots_adjust(left=0.35, right=0.95, top=0.92, bottom=0.15)

    plt.savefig(f"results/figures/{output_name}")
    plt.show()

if __name__ == "__main__":
    CSV_PATH = "results/metrics/best_golden_checkpoint_scan_1775603202.csv"
    # OOD
    # generate_dumbbell_plot(eval_data, CSV_PATH, "eval_ood_generalisation_gap.png",
    #     checkpoint_filter='best_eval',
    #     labels=('Evaluation', 'Golden Set'),
    #     title='OOD Evaluation vs. Golden Set Performance',
    #     xlim=(0.2, 0.6), ylabel_x=-0.04, use_axis_transform=True)
    #ID
    generate_dumbbell_plot(id_data, "results/metrics/id-eval-golden_set_results_1775416357.csv",
        "id_ood_generalisation_gap.png")