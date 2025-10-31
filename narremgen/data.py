"""
narremgen.data
=============
Generation of CSV files for advice, mapping, and context.
"""

import os
import pandas as pd
from pathlib import Path
from openai import OpenAI, OpenAIError
from .utils import postprocess_csv_text_basic
from .core import safe_chat_completion
from .utils import save_output

def generate_advice(topic: str, client: OpenAI, n_advice: int = 20, 
                    model="gpt-4o-mini", output_dir="outputs",
                    verbose=False):
    """
    Generate a CSV table of advices for a given topic.

    Each advice includes:
    - Num (index)
    - Topic
    - Advice (short title)
    - Sentence (a spoken line by a character)

    Parameters
    ----------
    topic : str
        The theme for which advices are generated.
    client : openai.OpenAI
        Authenticated OpenAI client.
    n_advice : int, optional
        Number of advices to generate. Default is 20.
    model : str, optional
        Model to use for generation. Default is "gpt-4o-mini".
    output_dir : str, optional
        Directory to save the CSV. Default is "outputs".

    Returns
    -------
    tuple
        (path_to_csv, num_rows) where path_to_csv is the file path of the advice table,
        and num_rows is the number of advices generated.
    """
    prompt = f"""
    Generate {n_advice} safety or behavioral advices for the topic: {topic}.
    For each advice, include:
    - A short title (3–6 words)
    - A clear, simple sentence which is said by another character to the main one living the scene

    <Format>
    Output must be ONLY a valid CSV, without line without header and without line after the valid CSV contents.
    ⚠️ Output must be ONLY a valid CSV.
    ⚠️ Output must be WITHOUT EMPTY LINES.
    ⚠️ Output must use standard UTF-8 characters (no special symbols or emojis).
    ⚠️ No text, no explanation, no code blocks.
    ⚠️ NEVER insert any semicolons in CELL VALUE/CONTENT
    ⚠️ The CSV must contain exactly 4 columns as defined in the header.
    ⚠️ If your output contains extra separators (;) inside cells, replace them by commas or plain spaces.
    ⚠️ Do not generate any additional semicolons beyond the 4 expected columns, for each line/row <== NOT more than 3>.
    ⚠️ Inside cells of the csv after header: Never insert any semicolons which are used as csv separators
    ⚠️ Inside cells of the csv after header: Use only plain text separated by spaces or between parenthesis.
    ⚠️ The CSV must start immediately with this header:
    The CSV header is the first line of the generated output as followed, before the line for the values:
    Num;Topic;Advice;Sentence
    </Format>
    """
    text = safe_chat_completion(client, model, [{"role": "user", "content": prompt}], max_tokens=4000)
    if not text:
        if verbose: print("!!! No answer from remote model - skipping step.")
        return None, 0
    log_path = os.path.join(output_dir, f"bad_rows_{topic.replace(' ', '_')}_advice.log")
    text = postprocess_csv_text_basic(text,expected_fields=4,log_path=log_path,verbose=verbose)

    file, real_n = save_output(text, f"Advice_{topic.replace(' ', '_')}", output_dir)
    try:
        df = pd.read_csv(file, sep=";")
        if list(df.columns) != ["Num", "Topic", "Advice", "Sentence"]:
            if verbose: print("!! CSV header unexpected:", df.columns.tolist())
        real_n = len(df)
    except Exception:
        real_n = 0

    return file, real_n


def generate_mapping(advice_file: str, client: OpenAI, SN_file: str, DE_file: str, 
                     model="gpt-4o-mini", output_dir="outputs", verbose=False):
    """
    Generate a Mapping CSV that assigns SN/DE codes to each Advice row.

    This function reads the Advice CSV and the official reference tables for
    narrative structures (SN) and emotional dynamics (DE), then prompts a
    language model to produce a strict semicolon-separated CSV with three
    columns: `Num;Code_SN;Code_DE`. The prompt embeds both official tables
    to forbid invented codes and encourages structural diversity while keeping
    one row per `Num`. A lightweight post-processor repairs malformed lines
    (by preserving `Num` and padding missing fields). Downstream validation
    (e.g., `utils.validate_mapping`) should still be applied to enforce that
    all codes exist in the references.

    Parameters
    ----------
    advice_file : str
        Path to the Advice CSV to map from. Must be semicolon-separated and
        include at least the column `Num`.
    client : openai.OpenAI
        An authenticated OpenAI client.
    SN_file : str
        Path to the official SN reference CSV. Must contain a `Code` column
        and preferably `Name` and `Narrative_Structure` for clarity.
    DE_file : str
        Path to the official DE reference CSV. Must contain a `Code` column
        and preferably `Name` and `Emotional_Sequence` for clarity.
    model : str, optional
        The model name used for mapping generation (e.g., 'gpt-4o-mini' or a
        reasoning model). Default: 'gpt-4o-mini'.
    output_dir : str, optional
        Directory where the generated Mapping file will be written. Default:
        'outputs'.
    verbose : bool, optional
        If True, prints details about parsing, repairs, and any fallback to
        raw text.

    Returns
    -------
    tuple[str | None, int]
        A pair `(path, n_rows)` where:
        - `path` is the path to the written file (CSV when parseable, otherwise
        a `*_raw.txt` fallback),
        - `n_rows` is the number of parsed rows (0 when raw fallback is used).

    Notes
    -----
    - The function does not guarantee that all codes are valid; it only guides
    the model. Always run `utils.validate_mapping` afterward to check and
    normalize codes.
    - The output CSV is strictly expected to have columns: `Num;Code_SN;Code_DE`.
    """

    adv = pd.read_csv(advice_file, sep=";")
    prompt = f"""
    You are given a table of advices. For each advice of number Num, assign:
    - Num number as first column
    - A narrative structure (SN code, e.g., SN1, SN3c…) as second column
    - An emotional dynamic (DE code, e.g., DE1, DE7…) as third column
    <Rules FOR DIVERSITY>
    It is mandatory to follow this rules:
    - You are mapping narrative advices to corresponding SN (Narrative Structure)
      and DE (Emotional Dynamic) codes.
    - Adopt a *diverse and pedagogical perspective* on the theme.
    - Encourage variation and contrast between lines: 
      ->some mappings should emphasize introspection, others analysis, 
      ->some emotional, others explanatory or didactic.
      ->Avoid using the same SN or DE too often. 
      ->Across all items, aim for a wide variety of narrative and emotional pairings
      Remember: diversity is part of the evaluation criteria of this mapping task.
    </Rules FOR DIVERSITY>
    <Format>
    Output must be ONLY a valid CSV, without line without header and without line after the valid CSV contents.
    ⚠️ The CSV must contain exactly 3 columns (for column names= Num;Code_SN;Code_DE), no more no less.
    ⚠️ Only use SN and DE codes from the provided reference lists.
    ⚠️ Do not invent new codes for SN et DE: SELECT ONLY CODES AVAILABLE.
    ⚠️ Output must be ONLY a valid CSV.
    ⚠️ Output must be WITHOUT EMPTY LINES.
    ⚠️ Output must use standard UTF-8 characters (no special symbols or emojis).
    ⚠️ If you need to separate items inside a cell, use character '_'  or spaces ' ' ONLY, AND NEVER ";" or ",".
    ⚠️ Inside cells of the csv after header: Never insert any semicolons which are used as csv separators
    ⚠️ Inside cells of the csv after header: Use only plain text separated by spaces or between parenthesis.
    ⚠️ The CSV must start immediately with this header:
    The CSV header is the first line of the generated output as followed, before the line for the values:
    Num;Code_SN;Code_DE
    </Format>

    <OFFICIAL_CODES_SN_DE>
    The EXISTING AND ONLY ALLOWED SN and DE are defined as below.
    """
    sn_ref = pd.read_csv(SN_file, sep=";")
    de_ref = pd.read_csv(DE_file, sep=";")
    
    prompt += "\n"
    prompt += "\n<OFFICIAL_SN_CODES>\n"
    prompt += "\nHere just below will be a table which gives the SN codes, the SN names and the SN structure definition.\n"
    prompt += "\nThe full list of ONLY ALLOWED SN codes is to choose from:\n"    
    prompt += sn_ref.to_csv(sep=";", index=False)
    prompt += "\n</OFFICIAL_SN_CODES>\n"
    prompt += "\n"
    prompt += "\n<OFFICIAL_DE_CODES>\n"
    prompt += "\nHere just below will be a table which gives the DE codes, the DE names and the DE structure definition.\n"
    prompt += "\nThe full list of ONLY ALLOWED DE codes is to choose from:\n"    
    prompt += de_ref.to_csv(sep=";", index=False)
    prompt += "\n</OFFICIAL_DE_CODES>\n"
    prompt += "\n"
    prompt += "\nWhen choosing a SN code and DE code, think enough deeply by reading the name and the definition to help the best selection for the advice.\n"
    prompt += "\n⚠️ You must ONLY select codes from the official list below. \n"
    # prompt += "\n⚠️ If you cannot find a suitable code, choose SN1 and DE1 by default. \n"
    prompt += "\n⚠️ If you cannot find a suitable code, choose the closest allowed DE with suitable meaning; avoid defaulting to DE1 or any other DE.\n"
    prompt += "\n⚠️ Never invent new codes (like SNk or DEk where k is not relevant).\n"
    prompt += "</OFFICIAL_CODES_SN_DE>"

    text = safe_chat_completion(client, model, 
                                [{"role": "user", 
                                  "content": prompt + "\n\n" + adv.to_csv(sep=';', index=False)}])
    if not text:
        if verbose: print("!!! No answer from remote model - skipping step.")
        return None, 0
    
    log_path = os.path.join(
        output_dir,
        f"bad_rows_{Path(advice_file).stem.replace('Advice', 'Mapping')}.log"
    )
    text = postprocess_csv_text_basic(text,expected_fields=3,log_path=log_path,verbose=verbose)
    
    file, real_n = save_output(
        text,
        os.path.basename(advice_file).replace("Advice", "Mapping").replace(".csv", ""),
        output_dir
    )
    try:
        df = pd.read_csv(file, sep=";")
        if list(df.columns) != ["Num", "Code_SN", "Code_DE"]:
            if verbose: print("!! Mapping CSV header unexpected:", df.columns.tolist())
        real_n = len(df)
    except Exception as e:
        if verbose: print(f"!! Failed to parse mapping file {file}: {e}")
        real_n = 0

    return file, real_n

def generate_context(advice_file: str, client: OpenAI, 
                     model="gpt-4o-mini", output_dir="outputs",
                     verbose=False):
    """
    Generate a CSV file with narrative context details for each advice.

    For each advice (Num), the function generates:
    - Character (age, role, gender, without first name)
    - Presence (who is around, with explicit speaker)
    - Location (setting)
    - Sensation (noise, smell, light, etc.)
    - Time (daytime)
    - Moment (mood/atmosphere keyword)
    - First_Name (plausible name for the character)

    Parameters
    ----------
    advice_file : str
        Path to the advice CSV file.
    client : openai.OpenAI
        Authenticated OpenAI client.
    model : str, optional
        Model to use for generation. Default is "gpt-4o-mini".
    output_dir : str, optional
        Directory where the context file will be saved. Default is "outputs".

    Returns
    -------
    tuple
        (path_to_csv, num_rows) where path_to_csv is the context file path,
        and num_rows is the number of rows generated.
    """

    adv = pd.read_csv(advice_file, sep=";")
    prompt = f"""
    For each advice in the table with columns (CSV), generate narrative context details:
    <Contents>
    - Character (age, role, gender) mais SANS PRENOM (aka First_Name), pour ne pas risquer d'incohérence avec colonne ci-après
    - Presence = all the people around who are present around the Character and between parenthesis explicitly who speaks the advice. Examples:
        "crow (alone, aka inner voice)"
        "street walkers (with an older woman who advises)"
        "store byers (with his daughter, who gives the advice)"
        "teacher classmates (with a classmat, who speaks)"
        "people in street (with near/aside/behind/afront a stranger who speaks the advice)"
        "someone at a window (with near/aside/behind/afront a stranger who speaks the advice)"
        "people on the sidewalk (with near/aside/behind/afront a stranger who speaks the advice)"
    - Location (urban, school, park…)
    - Sensation (noise, smell, light…)
    - Time (morning, afternoon…)
    - Moment (mood ambiance in one word like among [clear,sunny,cloudy,bright,calm,cool,warm,windy,quiet,noisy,soft,vivid,lively,nighty,peaceful,saturared] )
    - First_Name (plausible)
    </Contents>

    <Format>
    Output must be ONLY a valid CSV, without line without header and without line after the valid CSV contents.
    ⚠️ The CSV must contain exactly 8 columns (for column names= Num;Character;Presence;Location;Sensation;Time;Moment;First_Name), no more no less.
    ⚠️ Output must be ONLY a valid CSV.
    ⚠️ Output must use standard UTF-8 characters (no special symbols or emojis).
    ⚠️ Output must be WITHOUT EMPTY LINES.
    ⚠️ Output must be WITHOUT additional text, WITHOUT explanation, WITHOUT code blocks.
    ⚠️ Never use semicolons inside the cells (they are only separators).
    ⚠️ If you need to separate items inside a cell, use character '_'  or spaces ' ' ONLY, AND NEVER ";" or ",".
    ⚠️ Inside cells of the csv after header: Never insert any semicolons which are used as csv separators
    ⚠️ Inside cells of the csv after header: Use only plain text separated by spaces or between parenthesis.
    ⚠️ The Character field is with three fields: precise age, precise role, and precise gender.
    ⚠️ The First_Name field must contain ONLY the given VALID name, without any description.
    ⚠️ The Presence field is as following ONLY FORMAT "<Group or Person around the Character> (with <Speaker> who advises/talks)" in two parts
    ⚠️ DIVERSITY RULES (MANDATORY) --
    • First_Name MUST be varied across all rows in the CSV. No repetition of the same name is allowed.
    • Each Character must have distinct gender/role/age combinations where possible.
    • If a name would repeat, choose another realistic one from common French or English first names.
    • Vary Location, Time and Moment to ensure each line feels unique.
    • Never reuse the same combination of Character + First_Name twice.
    ⚠️ The CSV must start immediately with this header:
    The CSV header is the first line of the generated output as followed, before the line for the values:
    Num;Character;Presence;Location;Sensation;Time;Moment;First_Name
    </Format>
    """
    text = safe_chat_completion(client, model, 
                                [{"role": "user", 
                                  "content": prompt + "\n\n" + adv.to_csv(sep=';', index=False)}]) 
    if not text:
        if verbose: print("!!! No answer from remote model - skipping step.")
        return None, 0
    log_path = os.path.join(output_dir,
    f"bad_rows_{Path(advice_file).stem.replace('Advice', 'Context')}.log")
    text = postprocess_csv_text_basic(text,expected_fields=8,log_path=log_path,verbose=verbose)   

    file, real_n = save_output(
        text,
        os.path.basename(advice_file).replace("Advice", "Context").replace(".csv", ""),
        output_dir
    )
    try:
        df = pd.read_csv(file, sep=";")
        expected_cols = ["Num", "Character", "Presence", "Location", "Sensation", "Time", "Moment", "First_Name"]
        if list(df.columns) != expected_cols:
            if verbose: print("!! Context CSV header unexpected:", df.columns.tolist())
        real_n = len(df)
    except Exception as e:
        if verbose: print(f"!! Failed to parse context file {file}: {e}")
        real_n = 0

    return file, real_n

