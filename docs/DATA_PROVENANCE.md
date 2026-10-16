# Data provenance

This file records where each corpus came from, how it was processed, and the ethics position of the study.

## Human participants

None. The study used no interviews, surveys, observations or questionnaires, and collected no personal data from anyone. All evidence is machine-generated output from running `ahrag.py` over public document corpora and one corpus written by the author.

## The data collection instrument

`ahrag.py` is the instrument. It builds each corpus, runs retrieval, builds the prompts, calls the generator, parses citations, computes every metric and runs the statistical tests.

The two system prompts, `SYSTEM_PLAIN` and `SYSTEM_CITED` (Section 13 of `ahrag.py`), differ only in the citation instruction and are constant across the study. The full user prompt sent for each question, including the retrieved passages, is stored verbatim in the `prompt` field of every record under `results/<dataset>/audit/`.

## FinanceBench (public benchmark)

- **Source:** `PatronusAI/financebench` on Hugging Face (Islam et al., 2023). Licence: CC BY-NC 4.0.
- **Files:** `data/financebench/corpus.jsonl` (144 evidence documents), `data/financebench/qa.jsonl` (150 questions).
- **Processing:** one document per unique evidence text. Where two questions cite the same SEC exhibit, identical passages are collapsed using a SHA-256 content hash, so retrieval is not penalised for returning an identical twin.
- **Limitation:** the corpus uses the evidence passages, not the full filings. This approximates an open-book setting.

## CUAD (public benchmark)

- **Source:** the Atticus Project repository, test split (`data.zip` → `test.json`, SQuAD format), Hendrycks et al. (2021). Licence: CC BY 4.0.
- **Files:** `data/cuad/corpus.jsonl` (all 102 test-split contracts), `data/cuad/qa.jsonl` (150 questions).
- **Processing:** the 150 questions are a seeded random sample (seed 42) of the 1,244 answerable questions, chosen to fit the API budget. `build_cuad(n_questions=None)` produces the full set.
- **Note:** the questions are CUAD's templated clause-extraction prompts and the answers are clause spans. Because the prompts do not name a contract, corpus-wide retrieval is a stress test rather than a fair comparison.
- **Contact details:** as filed with the SEC, some contracts include business contact details of the parties (signatory names, corporate emails, notice-clause phone numbers). These are pre-existing public content of an established benchmark, are redistributed unmodified, and are not extracted, analysed or reported anywhere in this study.

## Azure technical corpus (written for this study)

- **Source:** written and compiled by the author, summarising public Microsoft Azure, RAG-architecture and enterprise-AI operations documentation.
- **Files:** `data/azure/corpus.jsonl` (55 short documents), `data/azure/qa.jsonl` (50 manually verified question–answer pairs: 22 lexical, 28 semantic; 47 map to one gold document and 3 to two).
- **Role:** a robustness check only. It is not given evidential weight equal to the two published benchmarks.

## Sample corpus

`data/sample/` (6 documents, 6 questions) lets anyone validate the pipeline end to end with `--dry-run` at no cost. It contributes no results.

## Credentials

No API key, token or credential is stored in this repository. `.env.example` is a placeholder template and `.gitignore` excludes the real `.env` file.

## Traceability

Every number in the dissertation is traceable to `results/summary.csv`, `results/stats.csv`, `results/error_analysis_summary.csv` or `results/supplementary_checks.txt`, and every individual generation is recorded in the audit logs under `results/<dataset>/audit/`.
