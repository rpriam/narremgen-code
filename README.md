# <br>
# **<u>NarrEmGen: Narrative Generation Pipeline</u>**<br>

## Main modules

- `pipeline`: Main entry point for batch generation, filtering, renumbering, and stats.
- `core`: OpenAI interface, API management, and safe/robust text generation utilities.
- `data`: Data preparation and input handling for topic–advice–prompt-based generation.
- `utils`: Validation, logging, and helper functions for file and mapping control.
- `narratives`: Text post-processing, style control, and thematic structure.
- `analyzestats`: Statistical exploration, corpus comparison with SN/DE counts.
- `interface`: Optional Tkinter GUI for text segmentation and tagging (not generation).

## Key features

- Multi-batch pipeline for narrative generation using OpenAI models (`gpt-4o-mini`, `gpt-4o`, `o3`, etc.).
- Automatic topic and advice mapping, dataset preparation, and structural generation (SN/DE patterns).
- Filtering of the rows/batch for advice+sentence/context/mapping csv, renumbering, appending csv and texts.
- Ready-to-use structure for reproducible experiments in text generation and educational content synthesis.

Note: This package is provided *“as is”* for the research and educational purposes.  <br>
      The code was written/debogged in iterative way with help of gpt5 openai + vs code. <br>
      All texts generated are synthetic and intended for future experimentations only.

## Usage

```
pip install narremgen
```

```python
import narremgen
from narremgen import pipeline

run_pipeline(
    topic="Walking_in_the_city",
    output_dir="./outputs",
    assets_dir="./narremgen/settings",
    n_batches=2,
    n_per_batch=20,
    output_format="txt",
    verbose=False
)
```

## Examples of output datasets

| ID | Corpus | n_texts |
|----|---------|---------|
| D1 | Keeping_town_clean | 257 |
| D2 | Learning_mistakes | 279 |
| D3 | Learning_skills | 296 |
| D4 | Protecting_water_forests | 253 |
| D5 | Stay_healthy | 300 |
| D6 | Walk_city | 276 |
| D7 | Walk_dawn | 271 |
| D8 | Walk_rain | 296 |
| D9 | Walk_water | 213 |
| D10 | Walk_wild | 255 |

Each generated corpus is stored under `outputs/` in CSV and TXT format.  
The naming convention is: `outputs/<corpus_name>_1/` for its directory<br>
<br>
Each directory contains:
```
topic, advice, and mapping tables in csv format and generated texts
and two subdirectories containing generated batched texts + csv files
```

## Warning

Only informed users or trainers should use this system in practice.  <br>
Some advice may be missing or mistaken du to ia/programming.  <br>
In future automatic checkings may be implemented for end user.

## References

- Priam, R. (2025). *Narrative and Emotional Structures For Generation Of Short Texts For Advice.*, hal-05135171, 2025.

---

© 2025 - NarrEmGen Project.
