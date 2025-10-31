"""
narremgen.narratives
===================
Batch and full generation of narrative texts from SN/DE mappings.
"""

import os, re, time
import pandas as pd
from typing import List
from openai import OpenAI
from .core import generate_text, estimate_tokens


def generate_narratives_batch(
    client: OpenAI,
    mapping_file: str,
    sn_file: str,
    de_file: str,
    advice_file: str,
    context_file: str,
    style_file: str = "style.txt",
    plain_text: bool = True,
    examples_file: str = "examples.txt",
    start_line: int = 0,
    batch_size: int = 5,
    language: str = "en",
    model_name: str = "gpt-4o-mini",
    max_tokens: int | None = 8000,
    output_dir: str = "./outputs/",
    target_words: int = 400,
    extra_instructions: str = None,
    verbose: bool = False,
):
    """
    Generate a batch of micro-narratives from aligned Advice, Context, and Mapping data.

    This function constructs a complete "giga-prompt" containing style rules, examples,
    official SN/DE structures, and a selection of aligned data rows from the Advice,
    Context, and Mapping CSVs. It sends the prompt to a model through the OpenAI API,
    retrieves the generated text for a specified number of narratives, and saves the
    output in a `batch_*.txt` file within the target directory.

    Parameters
    ----------
    client : openai.OpenAI
        Authenticated OpenAI client used for text generation.
    mapping_file : str
        Path to the Mapping CSV containing `Num`, `Code_SN`, and `Code_DE`.
    sn_file : str
        Path to the SN reference CSV (must include columns `Code`, `Name`, and `Narrative_Structure`).
    de_file : str
        Path to the DE reference CSV (must include columns `Code`, `Name`, and `Emotional_Sequence`).
    advice_file : str
        Path to the Advice CSV containing `Num`, `Topic`, `Advice`, and `Sentence`.
    context_file : str
        Path to the Context CSV containing situational fields (e.g., `Character`, `Location`, etc.).
    style_file : str, optional
        Path to the text file defining stylistic and narrative constraints. Default `'style.txt'`.
    plain_text : bool, optional
        If True, enforces a single paragraph per narrative instead of sectioned output (default True).
    examples_file : str, optional
        Path to a file with few-shot examples for model conditioning. Default `'examples.txt'`.
    start_line : int, optional
        Row index in the mapping file where this batch begins (default 0).
    batch_size : int, optional
        Number of narratives to generate in this batch (default 5).
    language : str, optional
        Output language code for the narratives (e.g., `'en'`, `'fr'`). Default `'en'`.
    model_name : str, optional
        Model used for text generation (default `'gpt-4o-mini'`).
    max_tokens : int | None, optional
        Token limit for completion (default automatically derived from model family).
    output_dir : str, optional
        Directory where the batch output will be saved (default `'./outputs/'`).
    target_words : int, optional
        Target word count per narrative, used to calibrate prompt size (default 400).
    extra_instructions : str | None, optional
        Additional text appended to the style section for extra control.
    verbose : bool, optional
        If True, prints token estimation and retry information.

    Returns
    -------
    str
        Path to the created `batch_*.txt` file containing generated narratives.

    Notes
    -----
    - Each batch operates independently to reduce token load and improve reproducibility.
    - Token estimation via `core.estimate_tokens()` is heuristic and helps avoid model overflow.
    - The resulting text files can later be merged using `merge_batches()`.
    """


    os.makedirs(output_dir, exist_ok=True)
    
    mapping = pd.read_csv(mapping_file, sep=";")
    sn = pd.read_csv(sn_file, sep=";").set_index("Code")
    de = pd.read_csv(de_file, sep=";").set_index("Code")
    context = pd.read_csv(context_file, sep=";").set_index("Num")
    advice = pd.read_csv(advice_file, sep=";").set_index("Num")
    style_text = open(style_file, encoding="utf-8").read()
    examples_text = open(examples_file, encoding="utf-8").read()

    if plain_text:
        style_text += (
            f"\n\nIMPORTANT: Utilise le balisage pour guider la création du contenu, "
            "mais NE PAS afficher les balises, commentaires, ni explications. "
            "Produit uniquement le texte brut, sous forme d’un paragraphe unique par récit "
            f"({target_words-50}–{target_words+50} mots)."
        )
    else:
        style_text += (
            f"\n\nIMPORTANT: Chaque section doit respecter environ {target_words//5} mots, "
            f"pour un total de {target_words-50}–{target_words+50} mots par récit."
        )

    if extra_instructions and extra_instructions.strip():
        style_text += "\n\n" + extra_instructions
    
    sn_ref = pd.read_csv(sn_file, sep=";")
    de_ref = pd.read_csv(de_file, sep=";")

    style_text += "\n\n--- OFFICIAL SN CODES ---\n"
    style_text += sn_ref.to_csv(sep=";", index=False)

    style_text += "\n\n--- OFFICIAL DE CODES ---\n"
    style_text += de_ref.to_csv(sep=";", index=False)

    batch = mapping.iloc[start_line:start_line + batch_size]

    prompt_parts: List[str] = []
    prompt_parts.append(style_text)
    prompt_parts.append("\n--- EXAMPLES ---\n")
    prompt_parts.append(examples_text)
    prompt_parts.append(
        f"\nNow generate narratives in {language.upper()} for the following rows.\n"
        "!! Each narrative MUST strictly apply the provided SN (narrative structure) "
        "and DE (emotional sequence) for its line.\n"
        "!! Do not invent new structures; follow the definitions above.\n"
    )

    for _, row in batch.iterrows():
        num = row["Num"]
        sn_code, de_code = row["Code_SN"], row["Code_DE"]

        sn_name = sn.loc[sn_code, "Name"]
        sn_struct = sn.loc[sn_code, "Narrative_Structure"]
        de_name = de.loc[de_code, "Name"]
        de_struct = de.loc[de_code, "Emotional_Sequence"]

        adv = advice.loc[num]
        ctx = context.loc[num]

        line_block = f"""
                    --- LINE {num} ---
                    Number: {num}
                    Topic: {adv['Topic']}
                    Advice: {adv['Advice']}
                    Sentence: "{adv['Sentence']}"
                    SN {sn_code} – {sn_name} ({sn_struct})
                    DE {de_code} – {de_name} ({de_struct})
                    Context: {ctx['Character']}, {ctx['Presence']}, {ctx['Location']}, {ctx['Sensation']}, {ctx['Time']}, {ctx['Moment']}, {ctx['First_Name']}
                    """
        prompt_parts.append(line_block)

    giga_prompt = "\n".join(prompt_parts)

    total_estimate, prompt_tokens, expected_output_tokens, max_model_tokens, max_tokens = estimate_tokens(
        giga_prompt, batch_size, target_words=target_words, model_name=model_name, max_tokens=max_tokens
    )

    if total_estimate > max_tokens * 0.8:
        if verbose:
            print(f"!! Warning: batch_size={batch_size} risque de saturer {model_name}.")
            print(f"   Estimation : {total_estimate} tokens (prompt ~{prompt_tokens}, output ~{expected_output_tokens})")
            print(f"   Limite modèle : {max_model_tokens} tokens")
            print(f"   Limite chosen : {max_tokens} tokens")
            usage_pct = round(total_estimate / max_tokens * 100, 1)
            print(f"!! Warning: batch_size={batch_size} risque de saturer {model_name}. ({usage_pct}% de la limite)")

    topic_slug = os.path.splitext(os.path.basename(advice_file))[0].replace("Advice_", "")

    max_retries = 3
    result = None
    for attempt in range(1, max_retries + 1):
        result = generate_text(
            giga_prompt,
            client,
            model_name=model_name,
            max_tokens=max_tokens
        )

        if result is not None:
            if verbose:
                print(f"Generation succeeded at attempt {attempt} "
                    f"for lines {start_line+1}-{start_line+batch_size}")
            break
        else:
            if verbose:
                print(f"!! Attempt {attempt} failed for lines "
                    f"{start_line+1}-{start_line+batch_size}")

    if result is None:
        placeholder = "\n\n".join(
            f"[GENERATION_FAILED for line {i}]"
            for i in range(start_line+1, start_line+batch_size+1)
        )
        result = placeholder
        if verbose:
            print(f"!!! All {max_retries} attempts failed - writing placeholder text.")

    batch_dir = os.path.join(output_dir, "batches_text")
    os.makedirs(batch_dir, exist_ok=True)
    batch_file = os.path.join(
        batch_dir,
        f"batch_{topic_slug}_{start_line+1}_{start_line+batch_size}.txt"
    )
    with open(batch_file, "w", encoding="utf-8") as f:
        f.write(result)

    if verbose:
        print(f"[narratives] Batch stored in {batch_dir}")

    if verbose:
        print(f"Narratives generated for lines {start_line+1} to {start_line+batch_size}")
        print(f"Output saved: {batch_file}")

    return batch_file 


def generate_narratives(
    client,
    mapping_file: str,
    sn_file: str,
    de_file: str,
    advice_file: str,
    context_file: str,
    style_file: str = "style.txt",
    examples_file: str = "examples.txt",
    plain_text: bool = True,
    start_line: int = 0,
    end_line: int = 10,
    batch_size: int = 5,
    language: str = "fr",
    model_name: str = "gpt-4o-mini",
    max_tokens: int | None = 8000,
    output_dir: str = "./outputs/",
    final_output: str = "merged.txt",
    output_format: str = "txt",   # "txt", "md", "tex", "docx", "pdf", "append"
    header_style: str = "full",   # "full", "simple", "none"
    target_words: int = 400,
    extra_instructions: str | None = None,
    verbose: bool = False,
):
    """
    Generate narrative texts from synchronized Advice, Context, and Mapping CSV files.

    This function orchestrates the full narrative synthesis step. It reads the
    aligned datasets, builds multiple batch prompts, calls `generate_narratives_batch`
    to create micro-narratives in groups, and then merges all batch outputs into a
    single document. It can produce plain text, Markdown, LaTeX, DOCX, or PDF output.

    Parameters
    ----------
    client : openai.OpenAI
        Authenticated OpenAI client used for text generation.
    mapping_file : str
        Path to the Mapping CSV file containing `Num`, `Code_SN`, and `Code_DE` columns.
    sn_file : str
        Path to the official SN reference CSV with structural definitions.
    de_file : str
        Path to the official DE reference CSV with emotional trajectories.
    advice_file : str
        Path to the Advice CSV containing the thematic prompts.
    context_file : str
        Path to the Context CSV providing situational details.
    style_file : str, optional
        Path to the text file defining narrative style guidelines. Default `'style.txt'`.
    examples_file : str, optional
        Path to a file containing few-shot examples for model guidance. Default `'examples.txt'`.
    plain_text : bool, optional
        If True, each generated narrative is a single unformatted paragraph (default True).
    start_line : int, optional
        Index of the first mapping row to process. Default 0.
    end_line : int, optional
        Index of the last mapping row to process (exclusive). Default processes all rows.
    batch_size : int, optional
        Number of narratives generated per batch (default 5).
    language : str, optional
        Target output language code (e.g., `'en'`, `'fr'`). Default `'en'`.
    model_name : str, optional
        Model name to use for narrative generation (default `'gpt-4o-mini'`).
    max_tokens : int | None, optional
        Token cap per API request (default determined by model family).
    output_dir : str, optional
        Directory for storing batch files and the merged output. Default `'./outputs/'`.
    final_output : str, optional
        Filename for the final merged narrative file. Default `'merged.txt'`.
    output_format : {'txt','md','tex','docx','pdf','append'}, optional
        Desired format of the merged output file (default `'txt'`).
    header_style : {'full','simple','none'}, optional
        Controls retention of batch headers during merge (default `'full'`).
    target_words : int, optional
        Target number of words per narrative (default 400).
    extra_instructions : str | None, optional
        Optional extra instructions appended to the style charter.
    verbose : bool, optional
        If True, prints progress and merge diagnostics.

    Returns
    -------
    str
        Path to the final merged narrative file.

    Notes
    -----
    - All input CSVs must be synchronized and share identical `Num` columns.
    - The function manages batch segmentation and merging automatically.
    - Non-text formats (`docx`, `pdf`) require external libraries such as `python-docx` and `reportlab`.
    """

    os.makedirs(output_dir, exist_ok=True)
    batch_files = []
    for s in range(start_line, end_line, batch_size):
        e = min(s + batch_size, end_line)
        file = generate_narratives_batch(
            client, mapping_file, sn_file, de_file, advice_file, context_file,
            style_file=style_file,
            examples_file=examples_file,
            plain_text=plain_text,
            start_line=s,
            batch_size=e - s,
            language=language,
            model_name=model_name,
            max_tokens=max_tokens,
            output_dir=output_dir,
            target_words=target_words,
            extra_instructions=extra_instructions,
            verbose=verbose
        )
        batch_files.append(file)

    base_name, ext = os.path.splitext(final_output)
    if output_format == "append":
        if not ext:
            ext = ".txt"  
        final_path = os.path.join(output_dir, base_name + ext)
        with open(final_path, "a", encoding="utf-8") as outfile:
            for f in batch_files:
                text = open(f, encoding="utf-8").read()
                text = _filter_headers(text, header_style)
                outfile.write(text.strip() + "\n\n")
        if verbose:
            print(f"Final file updated by append : {final_path}")
        return final_path
    else:
        merged_filename = base_name if base_name else "merged_output"
        merged_path = merge_batches(
            output_dir=output_dir,
            merged_filename=merged_filename,   
            output_format=output_format,
            header_style=header_style
        )
        if verbose:
            print(f"Final file generated : {merged_path}")
        return merged_path

def merge_batches(output_dir="outputs",
                  merged_filename="merged_output",
                  output_format="txt",
                  header_style="full",
                  verbose=False) -> str:
    """
    Merge multiple batch output files into a single document in the chosen format.

    This function scans a directory for files matching the pattern `batch_*.txt`,
    concatenates them in numeric order, optionally cleans headers, and writes the result
    as a merged file. Supported output formats include plain text, Markdown, LaTeX, DOCX,
    and PDF. Non-text outputs require `python-docx` or `reportlab` to be installed.

    Parameters
    ----------
    output_dir : str, optional
        Directory containing batch text files (default `'./outputs/'`).
    merged_filename : str, optional
        Base filename (without extension) for the merged output (default `'merged'`).
    output_format : {'txt','md','tex','docx','pdf'}, optional
        Desired output format (default `'txt'`).
    header_style : {'full','simple','none'}, optional
        Policy for header retention and comment stripping (default `'full'`).

    Returns
    -------
    str
        Path to the merged output file saved on disk.

    Notes
    -----
    - The merging order is based on numeric sorting of filenames (e.g., batch_1, batch_2, ...).
    - The function automatically detects encoding and ensures UTF-8 output.
    - For DOCX and PDF exports, content is first converted to text before rendering.
    """

    import glob, os

    candidates = [
        os.path.join(output_dir, "batches_text"),
        output_dir,
    ]
    search_dir = next((d for d in candidates if os.path.isdir(d)), output_dir)

    patterns = [
        "batch_*.txt",
        "batch_Filtered*.txt",
        "batch_*Filtered*.txt",
    ]
    batch_files = []
    for pat in patterns:
        batch_files.extend(glob.glob(os.path.join(search_dir, pat)))
    batch_files = sorted(set(batch_files), key=_sort_key)

    if not batch_files:
        if verbose:
            print(f"!! No batch file found in {search_dir}")
        return os.path.join(search_dir, f"{merged_filename}.{output_format}")

    merged_path = os.path.join(output_dir, f"{merged_filename}.{output_format}")

    if output_format in ["txt", "md", "tex"]:
        with open(merged_path, "w", encoding="utf-8") as outfile:
            for path in batch_files:
                text = open(path, encoding="utf-8").read()
                text = _filter_headers(text, header_style)
                outfile.write(text.strip() + "\n\n")
        if verbose:
            print(f"Merged File stored : {merged_path}")
        return merged_path

    elif output_format == "docx":
        try:
            from docx import Document
        except ImportError:
            if verbose:
                print("!! Module 'python-docx' missing. Fallback in .txt")
            return merge_batches(output_dir, merged_filename, "txt", header_style)

        doc = Document()
        for path in batch_files:
            text = open(path, encoding="utf-8").read()
            text = _filter_headers(text, header_style)
            doc.add_paragraph(text.strip())
            doc.add_page_break()
        doc.save(merged_path)
        if verbose:
            print(f"Merged File DOCX stored : {merged_path}")
        return merged_path

    elif output_format == "pdf":
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
        except ImportError:
            if verbose:
                print("!! Module 'reportlab' missing. Fallback in .txt")
            return merge_batches(output_dir, merged_filename, "txt", header_style)

        doc = SimpleDocTemplate(merged_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        for path in batch_files:
            text = open(path, encoding="utf-8").read()
            text = _filter_headers(text, header_style)
            story.append(Paragraph(text.replace("\n", "<br/>"), styles["Normal"]))
            story.append(Spacer(1, 20))
        doc.build(story)
        if verbose:
            print(f"Merged file PDF stored : {merged_path}")
        return merged_path

    else:
        if verbose:
            print(f"!! Unsupported Format : {output_format}. Fallback in .txt")
        return merge_batches(output_dir, merged_filename, "txt", header_style)


def _filter_headers(text: str, header_style: str) -> str:
    """
    Clean or retain headers and comment lines from a generated text block.

    This internal helper is used during merging to adjust the visibility of batch headers,
    comments, or section markers. It removes or simplifies formatting elements based on
    the selected header style, making the final document uniform and readable.

    Parameters
    ----------
    text : str
        The text content to process (may include headers, comments, or LaTeX/Markdown markers).
    header_style : {'full','simple','none'}
        Controls how much of the header structure is retained:
        - `'full'`: keep all headers and comments.
        - `'simple'`: keep light section headers only (Markdown ### or LaTeX \\subsubsection*),
        remove comments and metadata lines.
        - `'none'`: remove all headers and comments entirely.

    Returns
    -------
    str
        The cleaned text block formatted according to the selected header style.

    Notes
    -----
    - This function is mainly used by `merge_batches()` before assembling the final document.
    - It is not intended for public use and assumes well-structured narrative output files.
    """

    if header_style == "simple":
        return "\n".join(
            line for line in text.splitlines()
            if line.startswith("\\subsubsection*") or line.startswith("###") or not line.startswith("%")
        )
    elif header_style == "none":
        return "\n".join(
            line for line in text.splitlines()
            if not line.startswith("\\") and not line.startswith("%") and not line.startswith("###")
        )
    return text


def _sort_key(filename):
    """
    Extract a numeric index from a batch filename to allow natural sorting.

    Parameters
    ----------
    filename : str
        The batch filename (expected format: "..._<num>_...").

    Returns
    -------
    int
        The numeric index if found, else 0.
    """
    
    m = re.search(r"_(\d+)_", filename)
    return int(m.group(1)) if m else 0
