# Reproducibility Documentation

This document records exactly how the research was conducted so a reader can retrace the investigation and land on the same sources and conclusions.

## Environment

- **Tool**: `aii-web-tools` skill (scripts: `aii_fast_web_search.py`, `aii_fast_web_fetch.py`)
- **Python interpreter**: `/Users/Shared/aii_data/.claude/skills/.ability_client_venv/bin/python`
- **Search modes used**: `general` (Exa/keyless engines, Serper fallback) and `scholarly` (OpenAlex + Crossref)
- **Date of research**: 2026-11-23
- **Workspace**: `/Volumes/aii-s20019/work/run_qZEiMJexNbcK/round-2/research-1/src/`

## Search Queries Executed (in order)

### Scholarly Searches (OpenAlex/Crossref via `--mode scholarly`)

1. `coffee cooling experiment temperature time data` — 10 results
2. `Newton law cooling coffee experimental data` — 10 results
3. `coffee cooling dataset time temperature measurements` — 10 results
4. `cooling coffee cup experiment published data` — 10 results
5. `coffee temperature decay experiment data 2020 2025` — 10 results
6. `"Journal of Physics Special Topics" coffee cooling` — 10 results
7. `"Journal of Physics Special Topics" Newton cooling coffee` — 10 results
8. `Ciência e Natura coffee cooling` — 10 results
9. `Numerical simulation of the temperature distribution of coffee stored in cooled and natural environments Ciência e Natura` — 3 results
10. `"Physics Education" coffee cooling` — 10 results
11. `Using the structure of the residuals to evaluate the goodness of fit a pedagogical case study with Newton's law of cooling` — 10 results (found Dunnett 2025)
12. `Ferreira 2024 coffee cooling International Communications in Heat and Mass Transfer` — 10 results
13. `coffee cooling International Communications in Heat and Mass Transfer evaporation` — 10 results
14. `Newton's law of cooling experiment set using Arduino temperature sensor 2021` — 10 results
15. `Cataldo coffee cooling Newton's law of cooling` — 9 results
16. `When Three Points Aren't Enough Parameter Estimation Newton's Law of Cooling CODEE Journal` — 1 result
17. `"European Journal of Physics" coffee cooling` — 10 results
18. `Tennøe coffee cup cooling 2018 model` — 3 results

### General Web Searches (Exa/keyless via `--mode general`)

1. `Newton law cooling coffee dataset CSV` — 10 results
2. `coffee cooling data time temperature CSV download` — 9 results
3. `coffee cooling experiment data spreadsheet` — 10 results
4. `coffee temperature cooling curve data` — 10 results
5. `cooling coffee cup data points` — 10 results
6. `CSUB Physics 222 cooling coffee` — 10 results
7. `caveofprogramming cooling.csv` — 8 results
8. `coffee cooling dataset site:edu` — 9 results
9. `coffee cooling experiment "temperature vs time" data` — 10 results
10. `Newton law cooling coffee "data" "time" "temperature"` — 10 results
11. `When to add milk coffee cooling experiment raw data` — 10 results
12. `coffee cooling experiment data GitHub dataset` — 10 results
13. `coffee cooling data time temperature table` — 10 results
14. `coffee cooling experiment data "ambient" "room temperature" dataset` — 10 results
15. `coffee cooling data time temperature table site:github.com` — 10 results
16. `Ferreira cooling coffee cup experiment 2024 temperature time` — 10 results
17. `Radiation in the coffee cup Physics Education 2024 data` — 10 results
18. `How a soup bowl and a coffee cup cool down Ferreira 2024 data temperature time` — 10 results

## Pages/PDFs Fetched (full content, not just snippets)

| # | URL | Tool | Status | Notes |
|---|-----|------|--------|-------|
| 1 | https://doi.org/10.5902/2179460x74111 | fetch | ✓ HTML | Ciência e Natura paper on coffee grain storage, not cup cooling |
| 2 | https://www.csub.edu/~adzyubenko/phys222/Lab_Coffee.pdf | fetch | ✓ PDF | CSUB lab handout — procedure only, no data |
| 3 | https://github.com/caveofprogramming/PythonCourse/raw/refs/heads/main/17-Regression/cooling.csv | fetch | ✓ Raw CSV | 71 time-temperature points, no metadata |
| 4 | https://archive.math.arizona.edu/dsl/bcoffee.htm | fetch | ✓ HTML | Lomen & Lovelock dataset index page |
| 5 | https://archive.math.arizona.edu/dsl/data/coffee.dta | fetch | ✓ Raw data | 11 points (0-10 min) |
| 6 | https://archive.math.arizona.edu/dsl/data/coffee.inf | fetch | ✓ Description | Ambient 24°C |
| 7 | https://server.math.umanitoba.ca/~coopers5/pastcourses_unl_materials/project1_coffee_data.pdf | fetch | ✓ PDF | Manitoba workshop data — 13 points, ambient 71°F |
| 8 | https://web.mae.ufl.edu/uhk/COOLING-COFFEE.pdf | fetch | ✓ PDF | Kurzweg 1988/2017 — graphs only, no raw data tables |
| 9 | https://jwilson.coe.uga.edu/emt668/EMAT6680.2002.Fall/Irwin/assign12/cooling.html | fetch | ✓ HTML | UGA boiling water — 31 points (0-30 min), ambient 71°F |
| 10 | https://bpb-us-e2.wpmucdn.com/sites.up.edu/dist/6/248/files/2018/09/07.pdf | fetch | ✓ PDF | UP.edu Cooling Coffee — 8 points (0-14 min), ambient inferred |
| 11 | https://files.eric.ed.gov/fulltext/ED521035.pdf | fetch | ✓ PDF | Widjaja 2013 MERGA — educational study, no raw data |
| 12 | https://github.com/petermattia/coffee-and-milk-cooling/raw/refs/heads/main/coffee_cup_cooling.ipynb | fetch | ✓ Notebook | Simulation notebook, not experimental data |
| 13 | https://gist.github.com/Protonk/971510 | fetch | ✓ HTML/Gist | French press data — 20 points (0-95 min), excellent metadata |
| 14 | https://gist.github.com/gadenbuie/9370271 | fetch | ✓ HTML/Gist | Shiny app for coffee-cream simulation |
| 15 | https://repositorio.ucp.pt/server/api/core/bitstreams/facfcba7-6c1c-4700-92ec-4d50ef8a2565/content | fetch | ✓ PDF | Ferreira 2024 "Radiation in the coffee cup" — figures only, no raw tables |
| 16 | https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=1128&context=codee | fetch | ✗ 403 | CODEE Journal paper (Condori 2026) — could not access PDF |
| 17 | https://doi.org/10.1088/1361-6404/ae0201 | fetch | ✗ 403 | Dunnett 2025 Eur. J. Phys — could not access |
| 18 | https://doi.org/10.1088/1742-6596/2145/1/012068 | fetch | ✗ 403 | Arduino Newton cooling (J. Phys. Conf. Ser.) — could not access |
| 19 | https://doi.org/10.1088/1361-6552/aeb84b | fetch | ✗ 403 | Benz 2026 "When to add milk?" — could not access |
| 20 | https://beta.iopscience.iop.org/article/10.1088/1361-6552/ad6ac9/pdf | fetch | ✗ 403 | Ferreira "How a soup bowl..." — could not access |
| 21 | http://people.se.cmich.edu/lapp1da/ewexternalfiles/lab%203-newtoncooling-calculus.pdf | fetch | ✓ PDF | CMU lab handout — procedure, no data |
| 22 | https://www.vernier.com/vernier-ideas/cooling-coffee/ | fetch | ✓ HTML | Vernier page — graph only, data file 403 |
| 23 | https://www.desy.de/~fangohr/teaching/py4cs2026/lab8.html | fetch | ✗ Timeout | DESY lab page — could not fetch |

## Sources Requiring Author Contact / Special Access

The following potentially valuable sources could not be fully assessed due to access restrictions:

1. **Dunnett et al. 2025** (Eur. J. Phys. 46 055104) — "Using the structure of the residuals to evaluate the goodness of fit: a pedagogical case study with Newton's law of cooling" — DOI: 10.1088/1361-6404/ae0201
2. **Benz 2026** (Physics Education 60 015011) — "When to add milk? Using a coffee cooling experiment to teach measurement device selection" — DOI: 10.1088/1361-6552/aeb84b
3. **Ferreira 2024/2025** multiple Physics Education papers — raw data only in figures
4. **Condori et al. 2026** (CODEE Journal) — "When Three Points Aren't Enough" — has supplementary Julia code but PDF blocked
5. **Arduino Newton cooling** (J. Phys. Conf. Ser. 2021) — DOI: 10.1088/1742-6596/2145/1/012068

## Scoring Methodology (Applied Consistently)

Each candidate scored 0-2 on six criteria (max 10):
- **Resolution ≥6 points in 0-30 min**: 2 = ≥6 pts in first 30 min, 1 = some pts but <6, 0 = no/insufficient
- **Ambient temperature reported**: 2 = explicitly stated, 1 = inferable, 0 = not available
- **Independent source**: 2 = different lab/country from existing 4, 1 = same country diff lab, 0 = same lab or unknown
- **Raw data accessible**: 2 = numbers in table/CSV/text, 1 = only in figures (digitizable), 0 = not accessible
- **Initial temperature reported**: 1 = yes, 0 = no
- **Cup geometry reported**: 1 = yes, 0 = no

Threshold for inclusion: **≥6 points**

## Existing 4 Datasets (from iteration 1 artifacts)

For reference, the 4 datasets already in the panel:
1. **Kurzweg 1988/2017** (UFL) — water & black coffee, graphs only, score 5
2. **Manitoba Workshop** — 13 pts, ambient 71°F, score 9 ✓
3. **Ferreira 2024 cup** — multiple cups, figures only, score 6
4. **Lomen & Lovelock** (Arizona) — 11 pts but only 0-10 min, score 7 (fails duration)

## Restoring Removed Files

No large files were created that need restoration. All research outputs are text/JSON in this workspace.

## How to Retrace

1. Set up the aii-web-tools skill environment:
   ```bash
   export SKILL_DIR="/Users/Shared/aii_data/.claude/skills/aii-web-tools"
   export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
   ```

2. Run any search query from the lists above:
   ```bash
   $PY "$SKILL_DIR/scripts/aii_fast_web_search.py" --query "YOUR QUERY" --mode scholarly --max-results 10
   $PY "$SKILL_DIR/scripts/aii_fast_web_search.py" --query "YOUR QUERY" --mode general --max-results 10
   ```

3. Fetch any URL:
   ```bash
   $PY "$SKILL_DIR/scripts/aii_fast_web_fetch.py" fetch --url "URL" --max-chars 15000
   ```

4. For PDFs, use `--char-offset` to page through long documents.

5. The search results are deterministic for a given query at a given time but may drift over weeks/months as search indexes update. The fetches above were performed on 2026-11-23.