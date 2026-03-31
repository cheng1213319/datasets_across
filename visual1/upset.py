import pandas as pd

df_processed = pd.read_csv("../dataset_1/database/drug_comb/drugcomb_cleaned_with_mean_int.csv")
df_processed.dropna(subset=['drug_row', 'drug_col', 'cell_line_name'], inplace=True)

import matplotlib.pyplot as plt
from upsetplot import UpSet, from_contents

SCI_STYLE = {
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 900,
    'savefig.format': 'tiff',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
}

def prepare_upset_data(df, filter_isolated=False):

    drug_combinations = df.groupby('study_name').apply(
        lambda g: set(g['drug_row'].dropna().astype(str)) | set(g['drug_col'].dropna().astype(str))
    ).to_dict()

    cell_line_combinations = df.groupby('study_name')['cell_line_name'].apply(
        lambda x: set(x.dropna().astype(str))
    ).to_dict()

    combo_combinations = df.groupby('study_name').apply(
        lambda g: set(zip(
            g['drug_row'].dropna().astype(str),
            g['drug_col'].dropna().astype(str),
            g['cell_line_name'].dropna().astype(str)
        ))
    ).to_dict()

    if filter_isolated:
        all_drugs = set()
        for drug_set in drug_combinations.values():
            all_drugs |= drug_set

        overlapping_drugs = set()
        drug_counts = {drug: 0 for drug in all_drugs}
        for drug_set in drug_combinations.values():
            for drug in drug_set:
                drug_counts[drug] += 1

        overlapping_drugs = {drug for drug, count in drug_counts.items() if count >= 2}

        filtered_drug_combinations = {}
        for study, drug_set in drug_combinations.items():
            filtered_drug_combinations[study] = drug_set & overlapping_drugs

        return filtered_drug_combinations, cell_line_combinations, combo_combinations

    return drug_combinations, cell_line_combinations, combo_combinations


def plot_full_width_upset(data, output_name, color='#1f77b4', min_subset_size=2):

    filtered_data = {k: v for k, v in data.items() if len(v) > 0}
    if not filtered_data:
        return

    width, height = 7.5, 4.2

    fig = plt.figure(figsize=(width, height))
    plt.rcParams.update(SCI_STYLE)

    upset = UpSet(
        from_contents(filtered_data),
        facecolor=color,
        shading_color='#e6f2ff',
        subset_size='count',
        show_counts=True,
        sort_by='cardinality',
        sort_categories_by='cardinality',
        element_size=23,
        min_subset_size=min_subset_size
    )

    axes_dict = upset.plot(fig=fig)

    if 'intersections' in axes_dict:
        ax = axes_dict['intersections']
        ax.grid(False)

    for ax_name, ax in axes_dict.items():
        ax.tick_params(labelsize=SCI_STYLE['xtick.labelsize'])
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=SCI_STYLE['axes.labelsize'])
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=SCI_STYLE['axes.labelsize'])

        if hasattr(ax, 'texts'):
            for text in ax.texts:
                text.set_fontsize(SCI_STYLE['xtick.labelsize'])
                if ax_name == 'intersections':
                    text.set_rotation(60)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(ax_name == 'shaded')
        ax.spines['bottom'].set_visible(ax_name == 'intersections')
        ax.spines['left'].set_visible(ax_name == 'intersections')

    plt.savefig(
        f"{output_name}.tiff",
        dpi=900,
        bbox_inches='tight',
        pad_inches=0.02
    )
    print(f"已保存: {output_name}.tiff")
    plt.close(fig)


def plot_half_width_upset(data, output_name, color='#1f77b4'):

    filtered_data = {k: v for k, v in data.items() if len(v) > 0}
    if not filtered_data:
        return

    width, height = 3.7, 4.2

    fig = plt.figure(figsize=(width, height))
    plt.rcParams.update(SCI_STYLE)

    upset = UpSet(
        from_contents(filtered_data),
        facecolor=color,
        shading_color='#e6f2ff',
        subset_size='count',
        show_counts=True,
        sort_by='cardinality',
        sort_categories_by='cardinality',
        element_size=18
    )

    axes_dict = upset.plot(fig=fig)

    if 'intersections' in axes_dict:
        ax = axes_dict['intersections']
        ax.grid(False)

    for ax_name, ax in axes_dict.items():
        ax.tick_params(labelsize=SCI_STYLE['xtick.labelsize'])
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=SCI_STYLE['axes.labelsize'])
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=SCI_STYLE['axes.labelsize'])

        if hasattr(ax, 'texts'):
            for text in ax.texts:
                text.set_fontsize(SCI_STYLE['xtick.labelsize'])
                if ax_name == 'intersections':
                    text.set_rotation(60)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(ax_name == 'shaded')
        ax.spines['bottom'].set_visible(ax_name == 'intersections')
        ax.spines['left'].set_visible(ax_name == 'intersections')

    plt.savefig(
        f"{output_name}.tiff",
        dpi=900,
        bbox_inches='tight',
        pad_inches=0.02
    )
    plt.close(fig)


def filter_large_studies(df, min_entries=2000):
    study_counts = df['study_name'].value_counts()
    large_studies = study_counts[study_counts > min_entries].index.tolist()
    return df[df['study_name'].isin(large_studies)]


if __name__ == "__main__":
    df_processed_large = filter_large_studies(df_processed)
    drug_processed, cell_processed, combo_processed = prepare_upset_data(
        df_processed_large,
        filter_isolated=True
    )

    plot_full_width_upset(
        drug_processed,
        "Drug_Overlaps_Processed",
        color='#1f77b4',
        min_subset_size=2
    )

    plot_half_width_upset(
        cell_processed,
        "Cell_Line_Overlaps_Processed",
        color='#1f77b4'
    )

    plot_half_width_upset(
        combo_processed,
        "Drug_Combo_Overlaps_Processed",
        color='#1f77b4'
    )