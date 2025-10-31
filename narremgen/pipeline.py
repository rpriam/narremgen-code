"""
narremgen.pipeline
=================
Full orchestration of the multi-batch generation pipeline.
"""

import os, re, time
import pandas as pd
from openai import OpenAI
from .core import get_openai_key
from .data import generate_advice, generate_mapping, generate_context
from .utils import safe_generate, validate_mapping, merge_and_filter
from .utils import renumerote_filtered, audit_filtered, build_csv_name
from .utils import quick_check_filtered
from .analyzestats import analyze_sn_de_distribution
from .narratives import generate_narratives

def run_pipeline(topic, output_dir="outputs", assets_dir=None, 
                 model_advice="gpt-4o-mini",
                 model_mapping="o3",
                 model_context="gpt-4o-mini",
                 model_narrative="gpt-4o-mini",
                 n_batches=3,
                 n_per_batch=20,                                 # >=20
                 output_format="docx", 
                 verbose=True):
    """
    Run the complete Narremgen pipeline: data generation, filtering, narrative synthesis, and analysis.

    This function performs the full end-to-end workflow for one topic. It:
    1. Creates a versioned subdirectory for the topic.
    2. Generates Advice, Context, and Mapping CSVs through multi-batch processing.
    3. Applies consistency checks, filtering, and renumbering.
    4. Generates narrative texts from the synchronized CSVs.
    5. Computes SN/DE distribution statistics and correspondence analysis plots.

    If any stage fails coherence verification, the process stops gracefully to
    prevent incoherent text generation.

    Parameters
    ----------
    topic : str
        Descriptive label for the generation theme (e.g., "Urban Walk").
    output_dir : str, optional
        Root output directory where topic subfolders will be created (default 'outputs').
    assets_dir : str | None
        Directory containing the SN/DE reference CSVs and optional prompt assets
        (`style.txt`, `examples.txt`). Default: None.
    model_advice : str, optional
        Model used for Advice generation (default 'gpt-4o-mini').
    model_mapping : str, optional
        Model used for Mapping (default 'o3').
    model_context : str, optional
        Model used for Context generation (default 'gpt-4o-mini').
    model_narrative : str, optional
        Model used for final narrative generation (default 'gpt-4o-mini').
    n_batches : int, optional
        Number of Advice/Context/Mapping batches to generate (default 3).
    n_per_batch : int, optional
        Number of rows per batch (default 20).
    verbose : bool, optional
        If True, prints progress information throughout the run.

    Returns
    -------
    str | tuple[None, None, None, int]
        - On success: path to the final merged narrative file (.docx or .txt).
        - On early failure: (None, None, None, 0) to indicate an aborted run.

    Notes
    -----
    - Each run is isolated in a unique versioned folder (e.g., `<topic>_1`, `<topic>_2`).
    - The final narrative output can be used directly for corpus creation or evaluation.
    - The function integrates validation and basic analytics automatically.
    """
    try:
        if not assets_dir or not os.path.isdir(assets_dir):
            raise ValueError(
                "assets_dir must be a valid directory containing SN.csv, DE.csv, style.txt, etc."
            )

        topic_sanitized = topic.replace(" ", "_")

        existing = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
            and re.match(rf"^{re.escape(topic_sanitized)}(_\d+)?$", d)
        ]

        if existing:
            indices = [
                int(re.search(r"_(\d+)$", d).group(1)) for d in existing if re.search(r"_(\d+)$", d)
            ]
            last_k = max(indices) if indices else 0
        else:
            last_k = 0

        new_k = last_k + 1
        topic_versioned = f"{topic_sanitized}_{new_k}"
        output_dir = os.path.join(output_dir, topic_versioned)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "latest.txt"), "w", encoding="utf-8") as f:
            f.write(topic_versioned)

        if verbose:
            print(f"New run for topic '{topic}' -> directory : {output_dir}")
            print(f"Models: advice={model_advice} | mapping={model_mapping} | context={model_context} | narrative={model_narrative}")

        api_key = get_openai_key(verbose)
        client = OpenAI(api_key=api_key)

        advice_file, mapping_file, context_file = \
        generate_allcvs_batched_and_merge_and_dedup(
            client=client, topic=topic, assets_dir=assets_dir, 
            n_batches=n_batches, n_per_batch=n_per_batch, 
            output_dir=output_dir, verbose=verbose,
            model_advice=model_advice,
            model_mapping=model_mapping,
            model_context=model_context)

        if not mapping_file or not os.path.exists(mapping_file):
            print("!! Mapping file missing or unreadable, aborting early.")
            return None
        try:
            n_batched = len(pd.read_csv(mapping_file, sep=";"))
        except Exception:
            print("!! Failed to read mapping_file, aborting pipeline.")
            return None

        if verbose:
            print(f"Final coherence check for {output_dir}")

        if not quick_check_filtered(topic, output_dir, verbose=verbose):
            print(f"!! Issue detected in {output_dir}: inconsistent or missing CSV files.")
            print("!!! Pipeline stopped to prevent incoherent generation.")
            return None, None, None, 0
        else:
            if verbose:
                print(f"Pipeline complete and verified for topic '{topic}'.")

        final_file = generate_narratives(
            client, advice_file=advice_file, context_file=context_file, mapping_file=mapping_file,
            sn_file=os.path.join(assets_dir, "SN.csv"), de_file=os.path.join(assets_dir, "DE.csv"),
            style_file=os.path.join(assets_dir, "style.txt"), examples_file=os.path.join(assets_dir, "examples.txt"),
            plain_text=True, start_line=0, end_line=n_batched, batch_size=5, language="en",
            model_name=model_narrative, output_dir=output_dir,
            final_output=f"merged_{topic}.{output_format}",
            output_format=output_format,
            header_style="full", verbose=verbose
        )

        analyze_sn_de_distribution(mapping_file, os.path.join(assets_dir,"SN.csv"), 
                                os.path.join(assets_dir,"DE.csv"), output_dir, 
                                topic, verbose=verbose)

    except Exception as e:
        import traceback
        warning_msg = "\n!!  Something went wrong in run_pipeline."
        print(warning_msg)
        print("Error type:", type(e).__name__)
        print("Message:", e)
        traceback.print_exc()
        log_path = os.path.join(output_dir, "error_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(warning_msg + "\n")
            traceback.print_exc(file=f)
        print(f"(Error logged in {log_path})")
        return None
    
    return final_file


def generate_all_csv_batches(client, topic, assets_dir,
                         n_batches=5, n_per_batch=20,
                         output_dir="outputs",
                         model_advice="gpt-4o-mini",
                         model_mapping="o3",
                         model_context="gpt-4o-mini",
                         verbose=True):
    """
    Generate Advice, Context, and Mapping CSVs across multiple batches with built-in validation.

    This function orchestrates the batch generation of the three foundational datasets:
    Advice, Context, and Mapping. For each batch, it performs the following sequence:
    1. Generate Advice entries based on the given topic.
    2. Generate the corresponding Context aligned by `Num`.
    3. Generate the Mapping table linking each Advice to a valid SN/DE pair,
    retrying up to three times if invalid SN/DE codes are detected.

    All batches are stored under a `batches` subdirectory. Once all batches are
    successfully generated, the function merges each family of CSVs, renumbers them
    globally, and writes `Advice_Merged_<topic>.csv`, `Mapping_Merged_<topic>.csv`,
    and `Context_Merged_<topic>.csv` into the main output directory.

    Parameters
    ----------
    client : openai.OpenAI
        Authenticated OpenAI client used for all text generations.
    topic : str
        The topic name or label used in prompts and filenames.
    assets_dir : str
        Directory containing the official SN.csv and DE.csv reference files.
    n_batches : int, optional
        Number of batches to generate (default 5).
    n_per_batch : int, optional
        Number of rows (entries) per batch (default 20).
    output_dir : str, optional
        Root output directory for storing generated and merged CSVs (default 'outputs').
    model_advice : str, optional
        Model used for generating Advice data (default 'gpt-4o-mini').
    model_mapping : str, optional
        Model used for Mapping generation (default 'o3' for reasoning).
    model_context : str, optional
        Model used for Context generation (default 'gpt-4o-mini').
    verbose : bool, optional
        If True, prints detailed progress logs for each batch and stage.

    Returns
    -------
    None
        The function performs writing operations but returns no explicit object.
        All generated CSVs are saved on disk.

    Notes
    -----
    - Validation of Mapping rows is performed via `utils.validate_mapping()` after each batch.
    - The process includes automatic retries for batches producing inconsistent SN/DE codes.
    - The merged CSVs at the end serve as inputs for filtering, renumbering,
    and narrative text generation.
    """

    os.makedirs(output_dir, exist_ok=True)
    SN_file = os.path.join(assets_dir, "SN.csv")
    DE_file = os.path.join(assets_dir, "DE.csv")
    
    batch_dir = os.path.join(output_dir, "batches_csv")
    os.makedirs(batch_dir, exist_ok=True)

    advice_paths, mapping_paths, context_paths = [], [], []
    failed_batches = []

    for i in range(n_batches):
        subtopic = f"{topic}_(batch_{i+1})"
        if verbose:
            print(f"\n=== Generation of batch {i+1}/{n_batches} ===")

        advice_file = None
        try:
            advice_file = safe_generate(
                generate_advice, "Advice", subtopic, client,
                expected_rows=n_per_batch, model=model_advice,
                output_dir=batch_dir, verbose=verbose
            )
        except RuntimeError:
            print(f"!!! Batch {i+1} abandoned (Advice failed after internal retries).")
            failed_batches.append((i+1, "Advice"))
            continue
        if not advice_file:
            if verbose:
                print(f"!!! Batch {i+1} abandoned (Advice KO after 3 tries).")
            failed_batches.append((i+1, "Advice"))
            continue
        advice_paths.append(advice_file)

        context_file = None
        try:
            context_file = safe_generate(
                generate_context, "Context", advice_file, client,
                expected_rows=n_per_batch, model=model_context,
                output_dir=batch_dir, verbose=verbose
            )
        except RuntimeError:
            print(f"!!! Batch {i+1} abandoned (Context failed after internal retries).")
            failed_batches.append((i+1, "Context"))
            continue
        if not context_file:
            if verbose:
                print(f"!!! Batch {i+1} abandoned (Context KO after 3 tries).")
            failed_batches.append((i+1, "Context"))
            continue
        context_paths.append(context_file)

        mapping_file = None
        success = False
        for attempt in range(1, 4):
            mapping_file = safe_generate(
                generate_mapping, "Mapping", advice_file, client,
                SN_file, DE_file,
                expected_rows=n_per_batch, model=model_mapping,
                output_dir=batch_dir, verbose=verbose
            )
            if not mapping_file:
                if verbose:
                    print(f"!! Mapping failed (try number {attempt}) - re-lauch Mapping...")
                continue

            n_invalid_sn, n_invalid_de = validate_mapping(mapping_file, SN_file, DE_file, verbose=verbose)
            if n_invalid_sn == 0 and n_invalid_de == 0:
                success = True
                mapping_paths.append(mapping_file)
                if verbose:
                    print(f"Mapping not valid (batch {i+1}) at try number {attempt}.")
                break
            else:
                if verbose:
                    print(f"!! Mapping not valid (batch {i+1}) : {n_invalid_sn} SN, {n_invalid_de} DE - re-launch Mapping...")

        if not success:
            if verbose:
                print(f"!!! Batch {i+1} abandoned (Mapping KO after 3 tries).")
            failed_batches.append((i+1, "Mapping"))
            continue

    if failed_batches:
        log_path = os.path.join(batch_dir, "failed_batches.log")
        with open(log_path, "w", encoding="utf-8") as f:
            for num, stage in failed_batches:
                f.write(f"{num}\t{stage}\n")
        print(f"\n !!  {len(failed_batches)} batch(s) have failed : {failed_batches}")
        print(f"Details stored in {log_path}")

    def list_batch_files(prefix):
        return sorted(
            os.path.join(batch_dir, f)
            for f in os.listdir(batch_dir)
            if f.startswith(prefix) and f.endswith(".csv")
        )

    adv_files = list_batch_files("Advice_")
    map_files = list_batch_files("Mapping_")
    ctx_files = list_batch_files("Context_")

    if verbose:
        print(f"\n Batch filed found : "
            f"{len(adv_files)} advices, {len(map_files)} mappings, {len(ctx_files)} contexts")

    if not adv_files or not map_files or not ctx_files:
        raise RuntimeError(f"No batch file found in {batch_dir}")

    all_adv = pd.concat([pd.read_csv(f, sep=";") for f in adv_files], ignore_index=True)
    all_map = pd.concat([pd.read_csv(f, sep=";") for f in map_files], ignore_index=True)
    all_ctx = pd.concat([pd.read_csv(f, sep=";") for f in ctx_files], ignore_index=True)

    if len({len(all_adv), len(all_map), len(all_ctx)}) != 1:
        print(f"!! Incoherence at post-merge : "
            f"adv={len(all_adv)}, map={len(all_map)}, ctx={len(all_ctx)}")
        min_len = min(len(all_adv), len(all_map), len(all_ctx))
        all_adv, all_map, all_ctx = all_adv.head(min_len), all_map.head(min_len), all_ctx.head(min_len)

    new_nums = range(1, len(all_adv) + 1)
    for df in (all_adv, all_map, all_ctx):
        if "Num" in df.columns:
            df["Num"] = new_nums

    all_adv.to_csv(os.path.join(output_dir, build_csv_name("advice", "Merged", topic)), sep=";", index=False)
    all_map.to_csv(os.path.join(output_dir, build_csv_name("mapping", "Merged", topic)), sep=";", index=False)
    all_ctx.to_csv(os.path.join(output_dir, build_csv_name("context", "Merged", topic)), sep=";", index=False)

    if verbose:
        print(f"Global merge done and re-numbered for topic='{topic}'.")
        print(f"-> advice={len(all_adv)}, mapping={len(all_map)}, context={len(all_ctx)}")


def generate_allcvs_batched_and_merge_and_dedup(
    client,
    topic,
    assets_dir,
    n_batches=5,
    n_per_batch=20,
    output_dir="outputs",
    model_advice="gpt-4o-mini",
    model_mapping="o3",
    model_context="gpt-4o-mini",
    verbose=True
):
    """
    Execute the full multi-batch generation pipeline with filtering and renumbering.

    This function serves as the backbone of the CSV preparation process. It calls
    `generate_all_csv_batches` to produce multiple batches of Advice, Context, and
    Mapping files, then applies sequential cleaning steps:
    1. Merge all per-batch CSVs into single datasets.
    2. Filter out any invalid or incomplete rows across all three CSVs.
    3. Renumber the remaining rows to maintain synchronized and contiguous `Num` values.
    4. Optionally audit the resulting files for structural integrity.

    The resulting synchronized CSVs are ready for narrative text generation.

    Parameters
    ----------
    client : openai.OpenAI
        Authenticated OpenAI client for all model-based generations.
    topic : str
        Human-readable topic name used for naming files and directories.
    assets_dir : str
        Directory containing the official SN.csv, DE.csv, style.txt, and examples.txt.
    n_batches : int, optional
        Number of batches to generate (default 5).
    n_per_batch : int, optional
        Number of rows per batch (default 20).
    output_dir : str, optional
        Main output directory where all generated and filtered CSVs will be stored.
    model_advice : str, optional
        Model used for generating Advice (default 'gpt-4o-mini').
    model_mapping : str, optional
        Model used for Mapping (default 'o3').
    model_context : str, optional
        Model used for Context (default 'gpt-4o-mini').
    verbose : bool, optional
        If True, prints progress messages and summary reports.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing paths to the final synchronized CSVs:
        (advice_path, mapping_path, context_path).

    Notes
    -----
    - Each CSV returned corresponds to the *FilteredRenumeroted* stage of the pipeline.
    - This function ensures structural and semantic consistency across all
    data before narrative generation begins.
    - It should be invoked before calling `run_pipeline` or `generate_narratives`.
    """

    if verbose:
        print("\n=== PIPELINE MULTI-BATCH NARREMGEN ===")
        print(f"Topic            : {topic}")
        print(f"Directory output : {os.path.abspath(output_dir)}")
        print(f"Batches          : {n_batches} x {n_per_batch} rows each")

    generate_all_csv_batches(
        client=client,
        topic=topic,
        assets_dir=assets_dir,
        n_batches=n_batches,
        n_per_batch=n_per_batch,
        output_dir=output_dir,
        model_advice=model_advice,
        model_mapping=model_mapping,
        model_context=model_context,
        verbose=verbose
    )

    merge_and_filter(topic=topic, output_dir=output_dir, verbose=verbose)
    renumerote_filtered(topic=topic, output_dir=output_dir, verbose=verbose)
    if verbose: audit_filtered(topic=topic, output_dir=output_dir)

    advice_path  = os.path.join(output_dir, build_csv_name("advice", "FilteredRenumeroted", topic))
    mapping_path = os.path.join(output_dir, build_csv_name("mapping", "FilteredRenumeroted", topic))
    context_path = os.path.join(output_dir, build_csv_name("context", "FilteredRenumeroted", topic))

    if verbose:
        print("\nPipeline multi-batch finished (re-numbered files ready).")
        print(f"Advice file found : {advice_path}")
        print(f"Mapping file found: {mapping_path}")
        print(f"Context file found: {context_path}")

    return advice_path, mapping_path, context_path
