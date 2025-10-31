"""
narremgen.analyzestats
=====================
Statistical analysis of SN/DE distributions and Correspondence Analysis.
"""


import pandas as pd
import matplotlib.pyplot as plt

def _compute_diversity_index(series: pd.Series, verbose: bool = False):
    """
    Compute a diversity index and Herfindahl concentration score for categorical data.

    This internal utility function takes a pandas Series representing categorical labels
    (e.g., SN or DE codes) and computes:
    1. The percentage of unique categories used.
    2. The Herfindahl concentration index, defined as the sum of squared frequency shares.

    Parameters
    ----------
    series : pandas.Series
        Series containing categorical identifiers (e.g., `Code_SN` or `Code_DE`).
    verbose : bool, optional
        If True, prints a short summary of the diversity and concentration results.

    Returns
    -------
    tuple[float, float]
        A tuple `(diversity_pct, concentration_index)` where:
        - `diversity_pct`: the ratio (in %) of unique categories to total items,
        - `concentration_index`: Herfindahl index measuring concentration of use.
    """
    counts = series.value_counts(normalize=True)
    diversity_pct = len(counts) / len(series) * 100 if len(series) else 0
    herfindahl = (counts ** 2).sum()
    if verbose:
        print(f"Diversity: {diversity_pct:.1f}%  |  Herfindahl Index: {herfindahl:.4f}")
    return diversity_pct, herfindahl


def _plot_distribution(counts: pd.Series, title: str, output_path: str | None = None,
                       show_plot: bool = False, save_plot: bool = True):
    """
    Plot and optionally save a bar chart of categorical frequency distributions.

    Parameters
    ----------
    counts : pandas.Series
        A Series with category labels as index and their corresponding frequencies as values.
    title : str
        Title of the plot, typically including the topic or code type (e.g., "SN Distribution").
    output_path : str | None, optional
        Path to save the PNG image. If None, the plot is not saved.
    show_plot : bool, optional
        If True, displays the plot interactively (default False).
    save_plot : bool, optional
        If True, saves the plot to disk as a PNG image (default True).

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure object.
    """
    if counts is None or counts.empty:
        print(f"[warn] Empty dataframe passed to _plot_distribution for {title}. Skipping plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title(title)
    ax.set_xlabel("Code")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_plot and output_path:
        plt.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)
    return fig


def _export_distribution_csv(counts: pd.Series, ref_df: pd.DataFrame, output_path: str, code_type: str):
    """
    Export a frequency distribution of codes to a CSV summary file.
    """

    counts = counts.reset_index()
    counts.columns = [f"Code_{code_type}", "Count"]

    df = pd.DataFrame({
        f"Code_{code_type}": counts[f"Code_{code_type}"],
        "Count": counts["Count"],
        "Percentage": (counts["Count"] / counts["Count"].sum() * 100).round(2)
    })

    if "Name" in ref_df.columns:
        df = df.merge(ref_df[["Code", "Name"]],
                      left_on=f"Code_{code_type}", right_on="Code", how="left")
        df = df.drop(columns=["Code"]).rename(columns={"Name": f"Name_{code_type}"})

    df.to_csv(output_path, index=False)
    return output_path


def _format_percentage(value: float | int, scale: bool = True) -> str:
    """
    Format numeric values as percentage strings with consistent precision.

    Parameters
    ----------
    value : float | int
        The numeric value to format as a percentage (e.g., 0.346 or 34.6).
    scale : bool, optional
        If True, the input is assumed to be a ratio (0–1) and multiplied by 100
        before formatting (default True).

    Returns
    -------
    str
        The formatted percentage string, always including one decimal place.
    """
    try:
        if scale:
            value = value * 100
        return f"{float(value):.1f}%"
    except Exception:
        return "0.0%"


def analyze_sn_de_distribution(mapping_file, sn_ref_file, de_ref_file,
                               output_dir="outputs", topic=None,
                               save_plot=True, show_plot=False, verbose=False):
    """
    Compute and visualize the distribution of SN (narrative structures) and DE (emotional dynamics) codes.

    This function loads the Mapping CSV file and merges it with the official SN and DE
    reference tables to obtain descriptive names and categories. It computes the relative
    frequency of each code, exports summary tables as CSV, and creates optional bar plots
    to visualize code usage across the dataset. It also calculates basic diversity metrics
    such as percentage of unique codes and Herfindahl concentration indices.

    Parameters
    ----------
    mapping_file : str
        Path to the Mapping CSV containing at least columns `Code_SN` and `Code_DE`.
    sn_ref_file : str
        Path to the official SN reference CSV (must include columns `Code` and `Name`).
    de_ref_file : str
        Path to the official DE reference CSV (must include columns `Code` and `Name`).
    output_dir : str, optional
        Directory where output CSVs and plots will be stored (default `'outputs'`).
    topic : str | None, optional
        Optional tag appended to output filenames and plot titles.
    save_plot : bool, optional
        If True, saves PNG bar plots for SN and DE distributions (default True).
    show_plot : bool, optional
        If True, displays plots interactively instead of just saving them (default False).
    verbose : bool, optional
        If True, prints code distribution summaries and diversity statistics.

    Returns
    -------
    dict
        Dictionary containing:
        {
            'SN_diversity_pct': float,
            'DE_diversity_pct': float,
            'SN_concentration_index': float,
            'DE_concentration_index': float,
            'sn_file': str,  
            'de_file': str   
        }

    Notes
    -----
    - The diversity percentage indicates how many unique codes are used relative to total possibilities.
    - The Herfindahl index measures concentration: higher values mean less diversity.
    - Generated CSVs can be reused for corpus-level comparisons or cross-topic studies.
    """
    df = pd.read_csv(mapping_file, sep=";")
    sn_ref = pd.read_csv(sn_ref_file, sep=";")
    de_ref = pd.read_csv(de_ref_file, sep=";")

    counts_sn = df["Code_SN"].value_counts()
    counts_de = df["Code_DE"].value_counts()

    if counts_sn is None or counts_sn.empty or counts_de is None or counts_de.empty:
        print("[warn] Empty SN/DE counts detected. Skipping distribution plots.")
        return
    
    div_sn, conc_sn = _compute_diversity_index(df["Code_SN"], verbose=verbose)
    div_de, conc_de = _compute_diversity_index(df["Code_DE"], verbose=verbose)

    topic_tag = f"_{topic}" if topic else ""
    sn_csv = f"{output_dir}/SN_Distribution{topic_tag}.csv"
    de_csv = f"{output_dir}/DE_Distribution{topic_tag}.csv"
    sn_plot = f"{output_dir}/SN_Distribution{topic_tag}.png"
    de_plot = f"{output_dir}/DE_Distribution{topic_tag}.png"

    _export_distribution_csv(counts_sn, sn_ref, sn_csv, "SN")
    _export_distribution_csv(counts_de, de_ref, de_csv, "DE")

    _plot_distribution(counts_sn, f"SN Distribution {topic or ''}", sn_plot,
                       show_plot=show_plot, save_plot=save_plot)
    _plot_distribution(counts_de, f"DE Distribution {topic or ''}", de_plot,
                       show_plot=show_plot, save_plot=save_plot)

    if verbose:
        print(f"SN diversity: {_format_percentage(div_sn)} "
              f"| DE diversity: {_format_percentage(div_de)}")

    return {
        "SN_diversity_pct": div_sn,
        "DE_diversity_pct": div_de,
        "SN_concentration_index": conc_sn,
        "DE_concentration_index": conc_de,
        "sn_file": sn_csv,
        "de_file": de_csv,
    }
