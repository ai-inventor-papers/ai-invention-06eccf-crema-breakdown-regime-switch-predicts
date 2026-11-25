# gen_demo_art_experiment_1 — report_results

> Phase: `gen_paper_repo` · `gen_demo_art`
> Run: `run_qZEiMJexNbcK` — Newton's Law of Cooling Deviates by 6°C After 30 Minutes of Coffee Cooling
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_demo_art_experiment_1` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-11-25 02:32:04 UTC

````
<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<task>
Convert this artifact's Python script into a demo notebook with MINIMAL changes to the original code.
Split into cells, add markdown explanations between sections, add a visualization cell at the end.
Output: mini_demo_data.json + code_demo.ipynb (notebook that loads data from GitHub URL)
</task>

<artifact_info>
id: art_hPG4i7tI7B4v
type: experiment
title: Five-Model Coffee Cooling Comparison
summary: >-
  This experiment implements a rigorous five-model comparison for coffee cooling temperature trajectories across four published
  datasets (Kurzweg 1988, Manitoba workshop, Ferreira 2024 cup, Lomen & Lovelock). Models tested: B0 (Newton single-exponential
  baseline, 2 parameters), B1 (piecewise two-exponential, 5 parameters), C1 (coupled mass-energy ODE with evaporation, 3 parameters),
  C2 (Bayesian change-point MAP estimate, 3 parameters), C3 (flux-crossover with evaporation suppression, 3 parameters, tau
  sweep 3-8 min). Each dataset is evaluated with four split ratios (70%, 75%, 80%, 85%) yielding 16 total examples. Metrics
  computed: BIC (n ln(RSS/n) + 2k), out-of-sample RMSE, and predicted time-to-drinkable (50°C crossing). Results: C3 wins
  BIC on 3/4 datasets at 80% split; C2 achieves lowest OOS RMSE on all 4; all models agree within ~1-2 min on drinkable time.
  Output conforms to exp_gen_sol_out schema with full/mini/preview variants. Figures rendered via aii-data-fig-gen. Independent
  re-derivation script confirms all headline numbers.
workspace_path: >-
  /Volumes/aii-s20010/work/run_qZEiMJexNbcK/3_invention_loop/iter_1/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json
- reproducibility.md
out_demo_files:
- path: method.py
  description: Research methodology implementation
</artifact_info>

<github_repo>
Repo URL: https://github.com/ai-inventor-papers/ai-invention-06eccf-crema-breakdown-regime-switch-predicts
Raw data URL: https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-06eccf-crema-breakdown-regime-switch-predicts/main/round-1/experiment-1/demo/mini_demo_data.json

URLs won't work yet — files pushed to GitHub AFTER notebook creation.
Use local fallback pattern so notebook works locally (now) and in Colab (after deployment).
</github_repo>

<data_file_sizes>
Data files come in three sizes:
- preview_*_out.json — READ THIS to inspect the data structure
- mini_*_out.json (~3 examples) — use for prototyping/testing
- full_*_out.json (complete) — use for the final production run. NEVER open it directly (too large to read into context). Instead, extract values programmatically with shell commands (e.g. grep) or a Python script (use aii-long-running-tasks skill for scripts).
</data_file_sizes>

<install_dependencies_pattern>
Follow the aii-colab skill exactly. It has the install cell pattern, pre-installed package list, numpy 2.0 compat shims, and all Colab-specific rules.
</install_dependencies_pattern>

<data_loading_pattern>
`mini_demo_data.json` = curated subset for the demo.
Use this pattern for Colab compatibility (GitHub URL with local fallback):
```python
GITHUB_DATA_URL = "https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-06eccf-crema-breakdown-regime-switch-predicts/main/round-1/experiment-1/demo/mini_demo_data.json"
import http.client
import json
import urllib.request
from pathlib import Path

def load_data():
    try:
        with urllib.request.urlopen(GITHUB_DATA_URL, timeout=15) as response:
            return json.loads(response.read().decode())
    # Network errors, timeouts, a placeholder URL, bad JSON, a cut-off response.
    except (OSError, ValueError, http.client.HTTPException) as e:
        print("GitHub data unavailable, using local mini_demo_data.json:", e)
    local = Path("mini_demo_data.json")
    if local.exists(): return json.loads(local.read_text())
    raise FileNotFoundError("Could not load mini_demo_data.json")
```
</data_loading_pattern>

<figure_style>
Every plot in every notebook of this run shares one style. Paste this block, verbatim, at the end
of the imports cell:
```python
import matplotlib.pyplot as plt

# One colour per role, the same in every notebook of this run and in the paper's figures.
COLOURS = {
    "method": "#0173B2",  # the paper's own method, model or condition
    "baseline": "#949494",  # the main baseline, control or reference
    "other_1": "#DE8F05",  # the first further comparator
    "other_2": "#029E73",  # the second
    "other_3": "#CC78BC",  # the third
    "other_4": "#CA9161",  # the fourth
    "other_5": "#56B4E9",  # the fifth
}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#CCCCCC",
    "grid.alpha": 0.6,
    "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(color=list(COLOURS.values())),
    "image.cmap": "cividis",
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "figure.dpi": 120,
})
```
Colour each series by its ROLE from `COLOURS` (the proposed method `COLOURS["method"]`, the main
baseline `COLOURS["baseline"]`, further comparators `other_1`, `other_2`, ... in order). Use the
default colormap (cividis) for magnitudes and "RdBu_r" for signed values. Set no colours, fonts,
sizes or style sheets of your own.
</figure_style>

<notebook_structure>
--- Setup ---
Cell 1 (markdown): Title, description, what this artifact does.
Cell 2 (code): Install dependencies — follow the aii-colab skill's install cell pattern exactly. Fill in all packages imported by the artifact's code.
Cell 3 (code): Imports — copy original import block as-is, plus any additional imports needed for the notebook (e.g. matplotlib for visualization), then the <figure_style> block.
Cell 4 (code): Data loading helper — use the <data_loading_pattern> above.
Cell 5 (code): `data = load_data()`

--- Config ---
Config cell (code): Define ALL tunable parameters (iterations, epochs, n_samples, hidden_size, etc.) as variables at the top of this cell. Start with the ABSOLUTE MINIMUM values — the smallest that produce any output at all (e.g. 1 iteration, 2 samples, smallest array size). These get gradually increased during testing — see TODOs.

--- Processing ---
Remaining cells: One code cell per logical section of the original script. Add a markdown cell BEFORE each code cell. Copy code as closely as possible, with these changes:
  1. Replace file paths to use the loaded `data` variable.
  2. Use the config variables from the config cell (NOT hardcoded values).
  3. Minimal fixes are allowed if something doesn't work in notebook context (e.g. adjusting paths, removing CLI args, fixing imports), but keep changes to the absolute minimum.

--- Results ---
Visualization cell (code): Print key results in a readable table, plot numeric data with matplotlib if appropriate, in the <figure_style> style.
</notebook_structure>

<priority>
WORKING > OPTIMIZED. A small-scale demo that runs correctly is the goal. Once the notebook passes with minimum config values, scale up only if time permits — do NOT spend multiple retries chasing larger parameters. If a working version exists, finish and move on.
</priority>

<max_notebook_total_runtime>600s (10 min)</max_notebook_total_runtime>

<test_environment>
To test-run the notebook in a clean environment (simulating Colab), use the disposable venv at `/Volumes/aii-s20010/work/run_qZEiMJexNbcK/.shared_cache/nb_test_envs/art_hPG4i7tI7B4v-bb6fda0d916e`. It is on this machine's local disk, outside your workspace, which is on a slow network volume: never create a venv inside your workspace.
```bash
UV_VENV_CLEAR=1 UV_CACHE_DIR=/Volumes/aii-s20010/work/run_qZEiMJexNbcK/.shared_cache/uv uv venv -q --seed --python 3.12 /Volumes/aii-s20010/work/run_qZEiMJexNbcK/.shared_cache/nb_test_envs/art_hPG4i7tI7B4v-bb6fda0d916e
UV_CACHE_DIR=/Volumes/aii-s20010/work/run_qZEiMJexNbcK/.shared_cache/uv uv pip install -q --python /Volumes/aii-s20010/work/run_qZEiMJexNbcK/.shared_cache/nb_test_envs/art_hPG4i7tI7B4v-bb6fda0d916e/bin/python jupyter ipykernel
/Volumes/aii-s20010/work/run_qZEiMJexNbcK/.shared_cache/nb_test_envs/art_hPG4i7tI7B4v-bb6fda0d916e/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 code_demo.ipynb --output code_demo.ipynb
```
The timeout is set to <max_notebook_total_runtime>. The entire notebook must finish within this time.
`UV_VENV_CLEAR=1` recreates the venv empty at the start of every test, so each test starts clean. Do NOT delete it yourself: the pipeline removes it after you finish. If you run a test in the background, wait for its result before starting the next test.

What happens: the venv starts with only pip, jupyter and ipykernel. When the notebook's install cell runs, `google.colab` is NOT in sys.modules, so ALL packages get installed — non-Colab packages unconditionally, and Colab packages (numpy, pandas, etc.) at Colab's exact versions via the guard block. The result mirrors Colab's environment as closely as possible. If a cell fails, fix the notebook and re-run.
</test_environment>

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.


<todos>
TODO 1. Read and STRICTLY follow these skills: aii-colab, aii-long-running-tasks.
TODO 2. Read demo file and relevant preview_* files (preview only). Understand script structure: imports, setup, processing, output. Identify ALL tunable parameters (iterations, epochs, n_samples, hidden_size, batch_size, etc.) — these go in the config cell.
TODO 3. Create `mini_demo_data.json`: curated subset from at most ONE dataset (no more than 100 diverse examples). CRITICAL: do NOT read/grep full output file — may crash. Use `head -c 5000` or stream first entries with Python to pick examples.
TODO 4. Create `code_demo.ipynb` via NotebookEdit following <notebook_structure>. Set ALL config parameters to ABSOLUTE MINIMUM values — the smallest that produce any output (e.g. 1 iteration, 2 samples, smallest array sizes). Test-run using <test_environment>. Fix all errors until it passes.
TODO 5. GRADUALLY SCALE (but don't overdo it): increase config params step by step (e.g. ~2x each round). After each increase: test-run, record runtime, fix errors. STOP SCALING as soon as results look meaningful — a working small-scale demo beats a failed large-scale one. If full original params fit within <max_notebook_total_runtime> (10% margin), use them. Otherwise keep whatever works and comment out the true original values. Do NOT spend more than 2-3 scaling rounds.
TODO 6. Verify: (1) code_demo.ipynb contains GITHUB_DATA_URL = "https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-06eccf-crema-breakdown-regime-switch-predicts/main/round-1/experiment-1/demo/mini_demo_data.json" exactly, (2) mini_demo_data.json exists, (3) uses GitHub URL pattern not just open('mini_demo_data.json').
</todos>

---

Output the result as JSON to: `/Volumes/aii-s20010/work/run_qZEiMJexNbcK/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_hPG4i7tI7B4v/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "DemoExpectedFiles": {
      "description": "Expected output files from code demo notebook generation.",
      "properties": {
        "notebook": {
          "default": "",
          "description": "Path to the generated demo notebook. Example: 'code_demo.ipynb'",
          "title": "Notebook",
          "type": "string"
        },
        "mini_data_file": {
          "default": "",
          "description": "Path to the mini demo data JSON (curated subset). Example: 'mini_demo_data.json'",
          "title": "Mini Data File",
          "type": "string"
        }
      },
      "title": "DemoExpectedFiles",
      "type": "object"
    }
  },
  "description": "Dataset/experiment/evaluation \u2192 Jupyter notebook demo.\n\nTitle and summary come from the parent artifact (gen_art step),\nnot from the demo agent. The agent only outputs expected files.",
  "properties": {
    "out_expected_files": {
      "$ref": "#/$defs/DemoExpectedFiles",
      "description": "All output files you created. Must include the demo notebook."
    }
  },
  "required": [],
  "title": "CodeDemo",
  "type": "object"
}
```

IMPORTANT: this task is NOT complete until `/Volumes/aii-s20010/work/run_qZEiMJexNbcK/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_hPG4i7tI7B4v/.sdk_openhands_agent_struct_out.json` exists and contains JSON matching the schema above.

How accurately does Newton's law of cooling predict the temperature of a cup of coffee over 30 minutes, compared with published measurements?
````

### [2] SYSTEM-USER prompt · 2026-11-25 02:32:20 UTC

```
continue where you left off — reuse any partial work already written to disk. Do NOT start over.
```

### [3] SYSTEM-USER prompt · 2026-11-25 02:48:01 UTC

````
<validation-feedback>
Attempt 1 failed validation.

The output file `/Volumes/aii-s20010/work/run_qZEiMJexNbcK/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_hPG4i7tI7B4v/.sdk_openhands_agent_struct_out.json` does not exist yet.



---

Output the result as JSON to: `/Volumes/aii-s20010/work/run_qZEiMJexNbcK/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_hPG4i7tI7B4v/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "DemoExpectedFiles": {
      "description": "Expected output files from code demo notebook generation.",
      "properties": {
        "notebook": {
          "default": "",
          "description": "Path to the generated demo notebook. Example: 'code_demo.ipynb'",
          "title": "Notebook",
          "type": "string"
        },
        "mini_data_file": {
          "default": "",
          "description": "Path to the mini demo data JSON (curated subset). Example: 'mini_demo_data.json'",
          "title": "Mini Data File",
          "type": "string"
        }
      },
      "title": "DemoExpectedFiles",
      "type": "object"
    }
  },
  "description": "Dataset/experiment/evaluation \u2192 Jupyter notebook demo.\n\nTitle and summary come from the parent artifact (gen_art step),\nnot from the demo agent. The agent only outputs expected files.",
  "properties": {
    "out_expected_files": {
      "$ref": "#/$defs/DemoExpectedFiles",
      "description": "All output files you created. Must include the demo notebook."
    }
  },
  "required": [],
  "title": "CodeDemo",
  "type": "object"
}
```

IMPORTANT: this task is NOT complete until `/Volumes/aii-s20010/work/run_qZEiMJexNbcK/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_hPG4i7tI7B4v/.sdk_openhands_agent_struct_out.json` exists and contains JSON matching the schema above.

Produce `/Volumes/aii-s20010/work/run_qZEiMJexNbcK/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_hPG4i7tI7B4v/.sdk_openhands_agent_struct_out.json` again so it contains corrected JSON that matches the schema. Do not invent new fields.
</validation-feedback>
````
