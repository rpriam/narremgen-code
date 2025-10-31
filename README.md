# <br>
# **<u> NarrEmGen – Narrative Generation and Analysis Pipeline </u>** <br>

## Main modules

- `pipeline`: Main entry point for batch generation, filtering, renumbering, and statistical analysis of narrative texts.
- `core`: OpenAI interface, API management, and safe text generation utilities.
- `data`: Data preparation and input handling for topic–advice–prompt-based generation.
- `narratives`: Text post-processing, style control, and thematic structure.
- `analyzestats`: Statistical exploration, corpus comparison, PCA, Zipf’s law checks.
- `utils`: Validation, logging, and helper functions for file and mapping control.
- `interface`: Optional Tkinter GUI for text segmentation and tagging.

## Key features

- Multi-batch pipeline for narrative generation using OpenAI models (`gpt-4o-mini`, `gpt-4o`, `o1`, etc.).
- Automatic topic and advice mapping, dataset preparation, and structural filtering (SN/DE patterns).
- Text validation, renumbering, and corpus-wide statistics (length, TTR, Zipf distribution).
- Integration of a lightweight graphical interface for manual editing and segmentation.
- Ready-to-use structure for reproducible experiments in text generation and educational content synthesis.

Note: This package is provided *“as is”* for research and educational purposes.  
All texts generated are synthetic and intended for experimentation only.

## Usage

```
pip install narremgen
```

```python
import narremgen
from narremgen import pipeline

pipeline.run_pipeline("datasets/topics.csv")
```

## Examples of output datasets

| ID | Corpus | n_texts |
|----|---------|---------|
| O1 | Keeping_town_clean | 257 |
| O2 | Learning_mistakes | 279 |
| O3 | Learning_skills | 296 |
| O4 | Protecting_water_forests | 253 |
| O5 | Stay_healthy | 300 |
| O6 | Walk_city | 276 |
| O7 | Walk_dawn | 271 |
| O8 | Walk_rain | 296 |
| O9 | Walk_water | 213 |
| O10 | Walk_wild | 255 |

Each generated corpus is stored under `outputs/` in CSV and TXT format.  
The naming convention is: `outputs/<corpus_name>_1/` for its directory
Each directory contains:
```
topic, advice, and mapping tables in csv format and generated texts
```

## To be added next

- Fine-grained evaluation of narrative coherence and sentiment flow.
- Automatic style control and readability scoring for more variability.
- Automatic generation of new SN/DE during the early csv processing.
- Cross-corpus alignment for hybrid generation models.
- JSON schema for corpus metadata and other IA/LMM API.
- Full statistical Analysis after generating the batches
- Improvement of the pipeline, for loop textual refinement.

## References

- Priam, R. (2025). *Narrative and Emotional Structures For Generation Of Short Texts For Advice.*, hal-05135171, 2025.

## Acknowledgments

It is thanked OpenAI for its IA help at generating, debugging and refining most of the code for the NarrEmGen pipeline.<br>

---

© 2025 — NarrEmGen Project.
