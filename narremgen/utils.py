"""
narremgen.utils
==============
Utility functions for file management, validation, and postprocessing.
"""

from io import StringIO
import os, re, time
import pandas as pd

def build_csv_name(kind: str, stage: str, topic: str) -> str:
    """
    Construct a standardized CSV filename following the Narremgen naming convention.

    This function creates coherent filenames for all CSV files used within the
    pipeline. The convention improves readability and ensures cross-module consistency.
    Typical outputs include:
    - `Advice_Merged_<topic>.csv`
    - `Mapping_Filtered_<topic>.csv`
    - `Context_FilteredRenumeroted_<topic>.csv`

    Parameters
    ----------
    kind : str
        The file category or dataset type, such as 'advice', 'mapping', or 'context'.
    stage : str
        The pipeline stage to include in the filename (e.g., 'Merged', 'Filtered', 'FilteredRenumeroted').
    topic : str
        The topic slug or sanitized identifier (spaces replaced by underscores).

    Returns
    -------
    str
        The fully formatted filename string, e.g. `'Advice_FilteredRenumeroted_urban_walk.csv'`.

    Notes
    -----
    - The returned name does not include the directory path; it must be joined manually
    with `os.path.join(output_dir, ...)` when saving.
    - This function guarantees consistent case and separator usage across all pipeline outputs.
    """

    return f"{kind.capitalize()}_{stage}_{topic}.csv"


def save_output(text: str, filename: str, output_dir: str = "outputs", 
                ext: str = "csv", verbose=False):
    """
    Save the raw model output to disk, preferring CSV format with a text fallback.

    This helper attempts to parse the model output string as a semicolon-separated CSV.
    If parsing succeeds, the data are written as a properly formatted CSV file.
    If parsing fails (malformed content or wrong delimiter), the raw text is saved instead
    under the same base filename with the suffix `_raw.txt`.

    Parameters
    ----------
    text : str
        The text returned by the model, expected to represent a CSV table.
    filename : str
        The base filename to use when saving the output (without extension).
    output_dir : str, optional
        Directory where the file will be saved. Default is `'outputs'`.
    ext : str, optional
        Extension for structured output (default `'csv'`).
    verbose : bool, optional
        If True, prints diagnostic information about parsing errors or fallbacks.

    Returns
    -------
    tuple[str | None, int]
        `(path_to_file, num_rows)` where:
        - `path_to_file` is the saved file path (either `.csv` or `_raw.txt`),
        - `num_rows` is the number of rows successfully parsed (0 if raw text fallback is used).

    Notes
    -----
    - This function guarantees that every model call produces a persistent file,
    even when CSV parsing fails.
    - It normalizes column headers by stripping leading and trailing spaces.
    """

    os.makedirs(output_dir, exist_ok=True)
    path_csv = os.path.join(output_dir, f"{filename}.{ext}")

    if text is None:
        if verbose: print("!! No texte generated - probably a previous API error.")
        return None, 0
    
    try:
        df = pd.read_csv(StringIO(text), sep=";")
        df.columns = df.columns.str.strip() 
        with open(path_csv, "w", encoding="utf-8") as f:
            df.to_csv(f, sep=";", index=False)
        return path_csv, len(df)
    except Exception as e:
        path_txt = os.path.join(output_dir, f"{filename}_raw.txt")
        with open(path_txt, "w", encoding="utf-8") as f:
            f.write(text)
        if verbose: print(f"!! Error CSV ({e}), stored brut in {path_txt}")
        return path_txt, 0


def safe_generate(generate_fn, label, *args, expected_rows, max_retries=3, verbose=True, **kwargs):
    """
    Execute a generation function with automatic retries and CSV validation.

    This utility wraps a generation function (e.g., `generate_advice`, `generate_context`,
    or `generate_mapping`) to ensure robust output. It retries failed or malformed attempts
    up to a configurable maximum and validates that the resulting file exists, parses as CSV,
    and contains the expected number of rows.

    Parameters
    ----------
    generate_fn : Callable[..., tuple[str | None, int]]
        The generation function to call. It must return `(path, n_rows)`.
    label : str
        A label identifying the current stage, such as 'Advice', 'Context', or 'Mapping'.
    *args :
        Positional arguments forwarded directly to `generate_fn`.
    expected_rows : int
        Expected number of rows in the resulting CSV file.
    max_retries : int, optional
        Number of retries allowed beyond the initial attempt (default 3).
    verbose : bool, optional
        If True, prints progress, validation messages, and parsing errors.
    **kwargs :
        Additional keyword arguments passed through to `generate_fn`.

    Returns
    -------
    str
        The path to the valid CSV file once a successful attempt completes.

    Raises
    ------
    RuntimeError
        If all attempts fail or the final file does not meet the validation criteria.

    Notes
    -----
    - Between retries, the function waits briefly to avoid API throttling.
    - A generation attempt is considered valid only if the resulting file exists
    and the CSV contains exactly `expected_rows` entries.
    """


    for attempt in range(1, max_retries + 2):
        try:
            file_path, n_rows = generate_fn(*args, **kwargs)
        except Exception as e:
            if verbose:
                print(f"!! {label}: error when calling API ({e}) - try number {attempt}/{max_retries}")
            time.sleep(1)
            continue 

        if not file_path or not os.path.exists(file_path):
            if verbose:
                print(f"!! {label}: no file generated (try number {attempt})")
            time.sleep(1)
            continue

        try:
            df = pd.read_csv(file_path, sep=";")
        except Exception as e:
            if verbose:
                print(f"!!! {label}: reading impossible ({e}) - try number {attempt}")
            time.sleep(1)
            continue

        if len(df) == expected_rows:
            if verbose:
                print(f"{label} OK ({len(df)} rows)")
            return file_path

        if verbose:
            print(f"!! {label}: {len(df)}/{expected_rows} rows - new try number ({attempt})")
        time.sleep(1)

    raise RuntimeError(f"!!! Failure in generation step {label} after {max_retries+1} tries.")


def postprocess_csv_text_basic(text: str, expected_fields: int, log_path: str = None, verbose: bool = True):
    """
    Repair malformed CSV lines in raw model output and preserve row numbering.

    This lightweight postprocessor ensures CSV consistency when the model output
    contains irregular row lengths or missing columns. Each invalid line is corrected
    by keeping the first token (usually `Num`) and padding the remaining fields with
    synthetic placeholders (`BAD_fieldX`). Optionally, all invalid lines are logged
    for inspection.

    Parameters
    ----------
    text : str
        The raw semicolon-separated text output produced by the model.
    expected_fields : int
        Number of fields expected per row.
    log_path : str | None, optional
        Path to write a text log of invalid lines. If None, no log is created.
    verbose : bool, optional
        If True, prints the number of invalid lines detected and repaired.

    Returns
    -------
    str
        The corrected CSV text string with consistent column counts across rows.

    Notes
    -----
    - This function ensures the minimal validity required to load the data with pandas.
    - Bad lines are replaced in-place with placeholder tokens so that all datasets
    remain structurally aligned for subsequent filtering and merging.
    """

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    fixed, bad_lines = [], []

    for i, line in enumerate(lines, 1):
        parts = line.split(";")
        if len(parts) != expected_fields:
            bad_lines.append((i, line))
            num_val = parts[0].strip() if parts else "MISSING"
            new_parts = [num_val] + [f"BAD_field{n}" for n in range(2, expected_fields + 1)]
            parts = new_parts
        fixed.append(";".join(parts))

    if log_path and bad_lines:
        with open(log_path, "w", encoding="utf-8") as f:
            for idx, raw in bad_lines:
                f.write(f"{idx}\t{raw}\n")
        if verbose:
            print(f"!! {len(bad_lines)} rows not valid logged in {log_path}")

    return "\n".join(fixed)


def merge_and_filter(output_dir="outputs", topic="DefaultTopic", verbose=True):
    """
    Merge all merged CSV files (Advice, Mapping, Context), filter invalid rows, and synchronize datasets.

    This function reads the intermediate `*_Merged_<topic>.csv` files produced by the
    multi-batch pipeline, concatenates each family (Advice, Mapping, Context), and removes
    rows containing placeholder markers (e.g., `BAD_fieldX`). Filtering is performed
    synchronously: a row is kept only if it appears valid in all three datasets and shares
    a matching `Num` identifier. Cleaned files are saved as `*_Filtered_<topic>.csv` in the
    same directory.

    Parameters
    ----------
    output_dir : str, optional
        Directory containing the merged CSV files. Default is `'outputs'`.
    topic : str, optional
        Topic label used for matching filenames (default `'DefaultTopic'`).
    verbose : bool, optional
        If True, prints the number of rows processed, filtered, and retained.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary containing the filtered DataFrames for each dataset:
        {
            'advice': <DataFrame of Advice>,
            'mapping': <DataFrame of Mapping>,
            'context': <DataFrame of Context>
        }

    Notes
    -----
    - The filtering step maintains alignment between datasets, ensuring that every row
    number (`Num`) corresponds across Advice, Mapping, and Context.
    - Filtering relies on detection of invalid tokens containing the pattern `BAD_`.
    - This function prepares the datasets for renumbering and narrative synthesis.
    """


    patterns = {
        "advice": "Advice_Merged",
        "context": "Context_Merged",
        "mapping": "Mapping_Merged",
    }
    merged = {}
    for key, prefix in patterns.items():
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                 if f.startswith(prefix) and f.endswith(".csv") and f.endswith(f"_{topic}.csv")]
        if not files:
            if verbose: print(f"!! No file found for {key}")
            continue
        dfs = [pd.read_csv(f, sep=";", dtype=str) for f in sorted(files)]
        df_all = pd.concat(dfs, ignore_index=True)
        merged[key] = df_all
        if verbose: print(f"{key} merged ({len(df_all)} rows)")
    num_sets = [set(df["Num"]) for df in merged.values()
                if "Num" in df.columns and not df.empty]
    nums_all = set.intersection(*num_sets) if num_sets else set()
    bad_pat = re.compile(r"\bBAD_", re.I)
    bad_nums = set()
    for key, df in merged.items():
        if "Num" not in df.columns: continue
        mask_bad = df.apply(lambda s: s.astype(str).str.contains(bad_pat, na=False)).any(axis=1)
        bad_nums |= set(df.loc[mask_bad, "Num"])
        if verbose and mask_bad.any():
            print(f"{key}: {mask_bad.sum()} rows BAD to remove")
    keep_nums = nums_all - bad_nums
    if verbose:
        print(f"Num kept = {len(keep_nums)} / {len(nums_all)} (synchronized exclusion: {len(bad_nums)})")
    filtered = {}
    for key, df in merged.items():
        if "Num" not in df.columns: continue
        df_f = df[df["Num"].isin(keep_nums)].copy()
        filt_path = os.path.join(output_dir, build_csv_name(key, "Filtered", topic))
        df_f.to_csv(filt_path, sep=";", index=False)
        filtered[key] = df_f
        if verbose: print(f"{key} filtered -> {filt_path} ({len(df_f)})")
    return filtered


def renumerote_filtered(topic: str, output_dir="outputs", verbose=True):
    """
    Renumber filtered CSV datasets synchronously to ensure contiguous row identifiers.

    After filtering, the remaining Advice, Mapping, and Context datasets may differ slightly
    in length. This function aligns them by truncating to the smallest dataset, then assigns
    a consistent contiguous numbering sequence starting from 1. The resulting synchronized
    datasets are saved as `*_FilteredRenumeroted_<topic>.csv`.

    Parameters
    ----------
    topic : str
        Topic label used to identify the corresponding Filtered CSV files.
    output_dir : str, optional
        Directory containing the Filtered CSVs. Default is `'outputs'`.
    verbose : bool, optional
        If True, prints output file paths, number of rows written, and summary status.

    Returns
    -------
    None
        The function performs file I/O operations but does not return any object.

    Notes
    -----
    - The renumbering guarantees that all datasets share identical and consecutive `Num` values.
    - This step is essential for narrative generation, which depends on index alignment.
    - No content beyond the `Num` column is modified; only numbering is adjusted.
    """


    keys = ["advice", "mapping", "context"]
    dfs = {}
    for key in keys:
        path = os.path.join(output_dir, build_csv_name(key, "Filtered", topic))
        if not os.path.exists(path):
            if verbose:
                print(f"!! Missing file for {key} : {path}")
            continue
        dfs[key] = pd.read_csv(path, sep=";")

    if not dfs:
        print("!!! No file Filtered found to be re-numbered.")
        return None

    n = min(len(df) for df in dfs.values())
    new_nums = range(1, n + 1)

    for key, df in dfs.items():
        if "Num" in df.columns:
            df = df.head(n).copy()
            df["Num"] = new_nums
            path_out = os.path.join(output_dir, build_csv_name(key, "FilteredRenumeroted", topic))
            df.to_csv(path_out, sep=";", index=False)
            if verbose:
                print(f"{key} re-numbered -> {path_out} ({n} lignes)")
    if verbose:
        print("Re-numbering synchronized ended.")


def audit_filtered(topic: str, output_dir="outputs"):
    """
    Perform a consistency audit on synchronized FilteredRenumeroted CSV datasets.

    This diagnostic function verifies the internal integrity of the final synchronized
    datasets (`Advice_FilteredRenumeroted_<topic>.csv`, `Mapping_FilteredRenumeroted_<topic>.csv`,
    `Context_FilteredRenumeroted_<topic>.csv`). It checks that:
    1. All three files exist.
    2. They contain the same number of rows.
    3. The `Num` column is continuous and free of duplicates.
    4. The `Num` sets are identical across all files.

    Parameters
    ----------
    topic : str
        The topic identifier corresponding to the files being audited.
    output_dir : str, optional
        Directory where the FilteredRenumeroted files are located. Default is `'outputs'`.

    Returns
    -------
    None
        Prints human-readable results to the console.

    Notes
    -----
    - The audit reports any discrepancies between the three CSVs and warns about missing files.
    - This check is meant for developer and analyst verification before launching narrative generation.
    """

    files = {k: os.path.join(output_dir, build_csv_name(k, "FilteredRenumeroted", topic)) 
             for k in ["advice","mapping","context"]}

    dfs = {k: pd.read_csv(p, sep=";") for k, p in files.items() if os.path.exists(p)}
    if len(dfs) < 3:
        print("!! Missing files :", [k for k in files if k not in dfs])
        return
    nrows = {k: len(df) for k, df in dfs.items()}
    print("number of rows :", nrows)
    for k, df in dfs.items():
        nums = df["Num"].tolist()
        if sorted(nums) != list(range(1, len(nums) + 1)):
            print(f"!! {k}: Num not continuous or duplicates.")
        else:
            print(f"{k}: Num 1..{len(nums)} consistent.")
    common = set.intersection(*(set(df["Num"]) for df in dfs.values()))
    if all(len(df) == len(common) for df in dfs.values()):
        print("All files are perfectly synchronized.")
    else:
        print("!! Alignment inconsistency between files.")


def validate_mapping(mapping_file: str, SN_file: str, DE_file: str, 
                     verbose: bool = False) -> str:
    """
    Validate Mapping file codes against official SN and DE reference tables.

    This function ensures that all `Code_SN` and `Code_DE` values in the Mapping CSV
    exist in the official reference files. Invalid codes are replaced with default
    fallbacks (`SN1` for SN, `DE1` for DE). The corrected mapping is saved in place,
    overwriting the original file.

    Parameters
    ----------
    mapping_file : str
        Path to the mapping CSV to validate. Must include columns `Code_SN` and `Code_DE`.
    SN_file : str
        Path to the official SN reference CSV. Must include at least a `Code` column.
    DE_file : str
        Path to the official DE reference CSV. Must include at least a `Code` column.
    verbose : bool, optional
        If True, prints the number of invalid codes found and corrections applied.

    Returns
    -------
    tuple[int, int]
        A tuple `(n_invalid_sn, n_invalid_de)` representing the number of invalid SN
        and DE codes that were replaced.

    Notes
    -----
    - This correction guarantees that all SN/DE codes downstream are valid and consistent.
    - The default replacement values (`SN1a`, `DE1`) correspond to safe, neutral structures.
    - The updated mapping file overwrites the original to simplify downstream use.
    """


    mapping = pd.read_csv(mapping_file, sep=";", encoding="utf-8")
    sn_ref = pd.read_csv(SN_file, sep=";", encoding="utf-8")["Code"].tolist()
    de_ref = pd.read_csv(DE_file, sep=";", encoding="utf-8")["Code"].tolist()

    invalid_sn = ~mapping["Code_SN"].isin(sn_ref)
    invalid_de = ~mapping["Code_DE"].isin(de_ref)

    n_invalid_sn = invalid_sn.sum()
    n_invalid_de = invalid_de.sum()

    if n_invalid_sn > 0:
        if verbose:
            print(f"!! {n_invalid_sn} codes SN not valid detected -> replaced by 'SN1'")
        mapping.loc[invalid_sn, "Code_SN"] = "SN1"

    if n_invalid_de > 0:
        if verbose:
            print(f"!! {n_invalid_de} codes DE not valid detected -> replaced by 'DE1'")
        mapping.loc[invalid_de, "Code_DE"] = "DE1"

    mapping.to_csv(mapping_file, sep=";", index=False, encoding="utf-8")

    if verbose and (n_invalid_sn == 0 and n_invalid_de == 0):
        print("All codes SN/DE from mapping are validated.")

    return n_invalid_sn, n_invalid_de


def quick_check_filtered(topic: str, output_dir="outputs", verbose=True) -> bool:
    """
    Perform a rapid structural consistency check on the final FilteredRenumeroted CSVs.

    This lightweight check verifies that the three synchronized datasets
    (Advice, Mapping, and Context) exist, share the same number of rows,
    and have identical contiguous `Num` sequences. It provides a quick way
    to confirm alignment before launching narrative generation.

    Parameters
    ----------
    topic : str
        Topic identifier used to locate the FilteredRenumeroted files.
    output_dir : str, optional
        Directory containing the synchronized CSV files. Default is `'outputs'`.
    verbose : bool, optional
        If True, prints row counts and alignment diagnostics.

    Returns
    -------
    bool
        True if all three files exist, contain the same number of rows, and have
        matching contiguous `Num` columns. False otherwise.

    Notes
    -----
    - This function is primarily used within the pipeline to abort early in case of misalignment.
    - It ensures that each `Num` corresponds to the same logical entry across datasets.
    """

    import pandas as pd, os
    from .utils import build_csv_name

    files = {
        k: os.path.join(output_dir, build_csv_name(k, "FilteredRenumeroted", topic))
        for k in ["advice", "mapping", "context"]
    }

    missing = [k for k, p in files.items() if not os.path.exists(p)]
    if missing:
        if verbose:
            print(f"!! Missing files for {', '.join(missing)} in {output_dir}")
        return False

    dfs = {k: pd.read_csv(p, sep=";") for k, p in files.items()}

    nrows = {k: len(df) for k, df in dfs.items()}
    if verbose:
        print(f"Number of rows : {nrows}")

    nset = set(nrows.values())
    if len(nset) != 1:
        if verbose:
            print(f"!! Incoherence in size : {nrows}")
        n_min = min(nset)
        if verbose:
            print(f"Truncation possible at {n_min} rows for homogeneity.")
        return False

    num_refs = dfs["advice"]["Num"].tolist()
    aligned = all(df["Num"].tolist() == num_refs for df in dfs.values())

    if not aligned:
        if verbose:
            print("!! Number not aligned or not continuous between files.")
        return False

    if verbose:
        print(f"All files FilteredRenumeroted have coherent ({len(num_refs)} rows).")

    return True
