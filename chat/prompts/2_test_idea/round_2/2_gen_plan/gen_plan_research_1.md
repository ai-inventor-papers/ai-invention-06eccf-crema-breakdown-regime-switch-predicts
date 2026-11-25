# gen_plan_research_1 — test_idea

> Phase: `invention_loop` · round 2 · `gen_plan`
> Run: `run_qZEiMJexNbcK` — Newton's Law of Cooling Deviates by 6°C After 30 Minutes of Coffee Cooling
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_plan_research_1` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-11-24 04:56:24 UTC

````
<hypothesis>
kind: hypothesis
title: Parsimonious Crema-Breakdown Switch Predicts Coffee Cooling
hypothesis: >-
  Newton's law of cooling fails for coffee because it conflates an early evaporation-dominated regime (suppressed by crema
  coverage) with a late convection-dominated regime. A flux-crossover switching model with the crema decay time constant tau
  FIXED from independent literature measurements (3-8 minutes, using the midpoint 5.5 min or a prior distribution) — rather
  than swept as a free parameter — achieves lower out-of-sample RMSE and better BIC than both Newton's law and free-transition
  models, by eliminating the BIC penalty that caused the previous iteration's best predictor to fail the model screen. The
  parsimonious model predicts the drinkable-temperature crossing (50°C) within ±1 minute on held-out data.
motivation: >-
  Published measurements (Dunnett et al. 2025; Ferreira et al. 2024) show structured residuals from Newton's law fits to cooling
  water/coffee, proving the single-exponential model is physically incomplete. Arii & Nishizawa (2017) demonstrated that crema
  quantitatively delays cooling, but no model exists that uses crema breakdown as the switching mechanism between evaporation-
  and convection-dominated regimes. This gap matters practically: the drinkable-temperature crossing (43-60°C) is where regime
  transition typically occurs, and current models mis-predict it by 2-5 minutes. A crema-breakdown switching criterion provides
  a testable, physically grounded improvement.
assumptions:
- >-
  Crema thickness decays monotonically and can be approximated by an exponential or power-law decay with measurable time constant
- >-
  Evaporative heat flux is proportional to exposed liquid surface area (inversely related to crema coverage)
- >-
  Convective heat transfer coefficient remains approximately constant after crema breakdown
- >-
  Published datasets (Manitoba workshop, UFL Kurzweg 2017) or newly generated data provide sufficient time resolution to identify
  the transition point
- >-
  Room temperature, humidity, and airflow are approximately constant during the 30-minute measurement
investigation_approach: >-
  1. Collect/extract published coffee cooling datasets with time-temperature measurements (Manitoba, UFL, and others). 2.
  Fit three models: (a) standard Newton single-exponential, (b) two-exponential piecewise with free transition time, (c) crema-breakdown
  switching model where transition time is predicted from independently measurable crema properties (initial thickness, decay
  rate, critical coverage fraction). 3. Compare models using out-of-sample RMSE on held-out data, AIC/BIC for model selection,
  and specifically error at drinkable-temperature crossing (50°C ± 5°C). 4. Sensitivity analysis: vary crema decay parameters
  to test robustness of switching criterion.
success_criteria: >-
  The crema-breakdown switching model must: (1) achieve lower out-of-sample RMSE than Newton's law and free-transition two-exponential
  models on ≥80% of test datasets, (2) predict drinkable-temperature crossing within ±1 minute (vs. ±3-5 minutes for Newton's
  law), and (3) have the inferred transition time correlate with independent crema lifetime measurements (if available) or
  show physically plausible crema decay parameters (time constant 3-8 minutes). Failure mode: if free-transition two-exponential
  matches or beats the physics-based switching model, the crema-breakdown mechanism is not the right switching criterion.
related_works:
- >-
  Dunnett, Le & Holmqvist (2025) - Structured residuals in Newton's law fits to cooling water prove model incompleteness,
  but do not identify physical mechanism for regime transition
- >-
  Ferreira (2024, 2025) - Quantifies evaporative vs convective vs radiative contributions in cups/bowls, estimates constant
  heat transfer coefficients, but does not model time-varying evaporation from crema breakdown
- >-
  Arii & Nishizawa (2017) - Shows espresso crema delays cooling quantitatively, but uses static foam properties, not dynamic
  breakdown as switching trigger
- >-
  Kurzweg (2017) - Measures coffee cooling with/without cream, finds near-exponential behavior with 'slight differences' that
  are actually the regime transition signature
- >-
  Fractional derivative models (2020-2026) - Capture memory effects empirically but lack physical switching mechanism tied
  to crema physics
inspiration: >-
  Cross-domain transfer from ecology (regime shifts in lake ecosystems when vegetation cover decays below critical threshold)
  and materials science (oxide layer breakdown changing corrosion regime). The crema layer is a dynamic surface phase that,
  like a passivation layer, suppresses the dominant heat loss mechanism (evaporation) until it fails. This is a Level 2 procedural
  transfer: the workflow of 'identify dynamic surface phase → model its decay → use failure threshold as regime switch' is
  standard in corrosion science but never applied to coffee cooling.
terms:
- term: Newton's law of cooling
  definition: >-
    dT/dt = -k(T - T_env), predicting single-exponential temperature decay with constant cooling coefficient k
- term: Crema
  definition: >-
    The emulsion foam layer on espresso/coffee, composed of CO2 bubbles stabilized by lipids and melanoidins, typically 1-3
    mm thick initially
- term: Evaporation-dominated regime
  definition: >-
    Early cooling phase where latent heat loss from water vapor leaving the surface exceeds convective/conductive losses;
    suppressed by crema coverage
- term: Convection-dominated regime
  definition: >-
    Late cooling phase where crema has broken down, exposing liquid surface; cooling follows Newton's law with constant k
- term: Regime transition point
  definition: >-
    Time when evaporative heat flux drops below convective heat flux, triggered by crema thickness decaying below critical
    coverage fraction
- term: Drinkable-temperature crossing
  definition: >-
    Time when coffee temperature reaches 50°C ± 5°C (122°F ± 9°F), the practical target for consumption
summary: >-
  Newton's law fails for coffee because it misses a crema-breakdown-driven regime switch from evaporation-dominated to convection-dominated
  cooling. A physics-based switching model using measurable crema decay parameters predicts the full trajectory and drinkable-temperature
  crossing more accurately than any single-exponential or free-transition model.
alternates:
- title: Coupled Mass-Energy Evaporation Model
  hypothesis: >-
    A coupled ODE system tracking both coffee temperature and mass loss (evaporation) with temperature-dependent evaporation
    coefficient predicts cooling trajectories more accurately than Newton's law, without requiring a discrete regime switch.
  why_it_could_win: >-
    If the regime transition is gradual rather than sharp, a smooth coupled model with physically derived evaporation coefficient
    (from Hertz-Knudsen equation modified by crema coverage fraction) may outperform piecewise models by avoiding discontinuity
    artifacts at the transition.
- title: Residual-Structure Change Point Detection
  hypothesis: >-
    The transition point between cooling regimes can be detected purely from the structure of Newton's law fit residuals (building
    on Dunnett et al. 2025), and this data-driven change point predicts drinkable-temperature crossing as well as any physics-based
    switching criterion.
  why_it_could_win: >-
    If crema properties are hard to measure in practice but residual structure reliably identifies the transition, a purely
    statistical change-point model (e.g., Bayesian online change point detection on residuals) could be more practical and
    equally predictive, making the physics-based mechanism unnecessary for prediction.
- title: Information-Theoretic Model Selection for Regime Count
  hypothesis: >-
    Minimum Description Length (MDL) or Bayesian Information Criterion (BIC) applied to coffee cooling data will favor a two-regime
    model over single-regime Newton's law, but will not favor three or more regimes, establishing that exactly one physically
    meaningful transition exists.
  why_it_could_win: >-
    If the data intrinsically supports exactly two regimes regardless of the switching mechanism, then the key scientific
    contribution is quantifying the evidence for regime transition itself, not the specific crema-breakdown mechanism. This
    would be a more robust finding that doesn't depend on crema physics being correct.
_relation_rationale: >-
  Refines the crema-breakdown mechanism by fixing tau from physics to remove BIC penalty, same conceptual frame
_confidence_delta: decreased
_key_changes:
- >-
  Changed from tau sweep (3-8 min, 1 extra parameter) to FIXED tau from literature (0 extra parameters vs Newton), making
  the model parsimonious
- >-
  Shifted success criterion from 'wins pre-registered screen' to 'beats Newton on both RMSE and BIC' since the screen was
  underpowered at N=3
- >-
  Explicitly targets Newton's law accuracy quantification as a deliverable (the user's core question)
- >-
  Added bootstrap confidence intervals on full 30-min trajectory error, not just crossing time
- >-
  Dropped the claim that crema-breakdown is uniquely identified; the parsimonious model is a candidate mechanism to test,
  not a unique explanation
_strands:
- artifact: art_hPG4i7tI7B4v
  state: lead
  why: >-
    C3 achieves best BIC on 3/4 datasets and best drinkable-time prediction, but tau MLE hits sweep boundary (8 min); physics-based
    switching not uniquely identified vs free-transition/changepoint
- artifact: art_Jn8Po12wOvcN
  state: 'null'
  why: >-
    Pre-registered survival rule yields no survivors at N=3 (degenerates to unanimity); methodological limitation, not scientific
    refutation
_evidence_state: lead
_move: deepen
_move_rationale: >-
  C3 is a genuine lead (best prediction, fails only on BIC penalty); fixing tau from physics removes penalty and tests if
  crema-breakdown survives fair comparison
_direction_plan: >-
  Slot 1 (main): DEEPEN — parsimonious flux-crossover with tau fixed at literature midpoint (5.5 min) and as prior; re-evaluate
  on all 4 datasets with bootstrap CIs on RMSE and crossing time. Slot 2 (secondary): COVER — compute Newton's law full 30-min
  trajectory error with bootstrap confidence bands. Slot 3 (secondary): WIDEN — search for 1-2 additional published datasets,
  add to panel, quick re-screen parsimonious C3 vs all candidates.
_coverage: partial
_coverage_statement: >-
  Next iteration will answer the user's core question (Newton's law 30-min accuracy with CIs) while testing whether a parsimonious
  crema-breakdown model beats Newton on both prediction and model selection criteria.
_candidates_considered: 6
relation_type: evolution
</hypothesis>

<available_domain_handbooks>
Domain handbooks below capture expert knowledge for a specific field — its landscape, prior work, dead ends, evaluation norms, and what counts as a genuinely novel contribution. If one is relevant to your research topic, READ that skill BEFORE proceeding; read the most relevant one(s), or none if none apply. When none fit, do not force one — instead ground your work harder in primary sources and hold novelty claims to extra scrutiny, since you have no curated map of this field's prior work and dead ends. Use it for the methods, proper baselines, and evaluation this field demands.

- **aii-handbook-auto-computational-linguistics** — Field handbook for computational linguistics as a SCIENCE of language — grammaticality and minimal pairs (BLiMP), surprisal versus reading times, linguistic structure in LMs, annotator disagreement an
- **aii-handbook-auto-mechanistic-interpretability** — Field handbook for mechanistic interpretability of neural networks — circuit discovery, activation and attribution patching, sparse autoencoders, transcoders, attribution graphs, steering vectors, pro
- **aii-handbook-auto-multi-agent-llm-systems** — Field handbook for multi-agent LLM systems (MAS) — orchestration topology, multi-agent debate, mixture-of-agents, verifier and critic agents, inter-agent protocols (MCP/A2A), failure attribution and s
- **aii-handbook-auto-neurosymbolic** — Field handbook for neuro-symbolic AI — text-to-logic autoformalization (NL to FOL), LLM-plus-solver and prover pipelines (Prolog, ASP, SMT), probabilistic-differentiable NeSy (DeepProbLog, Scallop), r
</available_domain_handbooks>

<strategy_domain_reasoning>
How the strategist established that researchers in this field reason, at the level of the field's principles and standards of evidence. Take it as the starting point for the concrete practice below — extend or correct it where your own reading disagrees, and say so when you do.

The field is classical convective heat-transfer measurement as practiced in physics-education and applied-thermodynamics journals (European Journal of Physics, Physics Education, International Communications in Heat and Mass Transfer). Field norms verified from primary sources: (1) The baseline is always a fitted Newton single-exponential, and its failure is quantified by residual structure (Dunnett et al. 2025, Eur. J. Phys. 46 055104), not just RMSE — structured residuals prove model incompleteness. (2) A competing model is accepted only if it beats Newton at equal or fewer free parameters, using BIC or AICc comparison per dataset, and only if out-of-sample prediction of the drinkable-temperature crossing (~50°C) improves. (3) Evaporation is the known first-order loss channel above ~60°C (Ferreira 2024; Int. Commun. Heat Mass Transfer 2025), so any two-regime model whose late regime drops the latent-heat term must be physically justified. (4) 'Crema coverage' is a dynamic quantity no published cooling dataset measures, so the hypothesis's literal switching criterion is untestable on existing data and must be proxied by the evaporation-flux / convection-flux crossover computed from known h, latent heat L, and vapor-pressure coefficients — a physically-derived time that is a function of the temperature trajectory itself. Usual failure modes: overfitting a 5-parameter model to a 15-point dataset (BIC collapses to Newton), confounding 'beats Newton in-sample' with 'predicts the drinkable crossing out-of-sample' (the user's actual question), and using a single dataset so one lucky fit masquerades as a mechanism. The panel must have ≥3 independent published sources, each with its own fold, and the confirmation fold must never be touched by the screen.
</strategy_domain_reasoning>

<domain_practice>
FIRST WORK OUT HOW THIS KIND OF STUDY IS ACTUALLY BUILT IN THIS FIELD. Then
write the plan.

The strategy already settled what the field believes and what it counts as
convincing. Your job is the level below that: how work of exactly this kind
is designed, run and reported by the people who do it, concretely enough that
the executor's output would be recognised as competent by one of them.

Establish, for this field and this artifact type:

- BASELINES AND COMPARISONS. Which comparisons appear in every paper of this
  kind, named specifically. Which one would a reviewer name first if it were
  missing, and what is the standard way of tuning it fairly?
- CASES AND DATA. Which datasets, corpora, cohorts, benchmarks, case sets or
  sources are standard here, and which are known to be saturated, leaked,
  deprecated or unrepresentative. Prefer the ones the field actually uses,
  and say why when you pick something else.
- CONTROLS AND WHAT IS HELD CONSTANT. What has to be held fixed for the
  comparison to mean anything, and which confound this design is most likely
  to be caught on.
- HOW MUCH IS ENOUGH. Sample sizes, item counts, seeds, repeats, splits —
  the number below which nobody in this field believes a result, and what the
  field reports alongside a point estimate (variance, intervals, a
  significance or uncertainty treatment). When a power analysis, or the
  effect sizes already on record, say the panel cannot detect the size of
  effect the plan is chasing, the fix is more graded samples or checkpoints
  — not more candidate metrics. An underpowered panel stays underpowered no
  matter how many readouts run over it.
- MEASURES AND REPORTING CONVENTIONS. Which measures are standard, how they
  are computed here, and the conventions a reader will expect to see — what
  is reported, against what, in what form.

Not every axis applies to every artifact type: a proof has proof standards
and an accepted level of rigour rather than sample sizes, a research artifact
has source quality and coverage, a dataset has provenance, licensing and
documentation norms. Answer the ones that apply and skip the ones that do not
rather than inventing content for them.

HOW MUCH EFFORT. Bounded, like the strategist's: the fitting domain handbook
plus a handful of targeted lookups — one or two recent papers doing this exact
kind of study, a benchmark or dataset card, a methods or reproducibility note.
Read what the field does; do not reason it out from first principles. You
cannot run code, so this is reading only.

THEN CHECK THE PLAN AGAINST IT.
- `domain_practice`: what you established above, concretely — named baselines,
  named data, the numbers, the measures, with what you read.
- `practice_alignment`: go through the plan you just wrote against that list
  and say, point by point, where it MEETS the field's practice and where it
  DEPARTS from it. For every departure: why it is justified here (budget,
  scope, the claim being narrower) and what it costs the result's
  credibility. A departure nobody named is the one a reviewer finds.
- Where the check exposes a gap you can close inside the budget, close it in
  the plan rather than reporting it. `practice_alignment` is for what remains
  after you have fixed what you can.
</domain_practice>

<field_extras>
What the field's best-cited recent papers deliver beyond the minimum, as the report step found
it in the style exemplars. These are what lifts a paper above the field's floor, not
requirements.

- Complete raw data tables in appendix or supplement (for reproducibility; delivered in appendix; needs during the run: Save all raw measurement data, not just fitted parameters; seen in Dunnett et al. (2025); Cataldo et al. (2016))
- Code and analysis scripts released (for reproducibility; delivered in repo; needs during the run: Version-control analysis scripts; seen in Cataldo et al. (2016) - mentions LabVIEW/automation)

When this artifact tests a claim of an entry's claim kind, the plan collects what the
entry needs during the run (the saved values, the extra model run) inside this artifact's
own steps and budget, so the paper can deliver the extra later. It adds no new phase for it, and
when the extra and the measurement this artifact exists for do not both fit, the extra is dropped.
</field_extras>

<domain_vocabulary>
The names you give methods, conditions and metrics here travel into the report and the paper,
so they follow the paper's naming rules from the start:

- Name each concept, metric and condition the way the field names it. If the field has a word for
  what you are describing, that word is the one that goes in the document.
- Never invent a private label for something the field already names. A reader searching for the
  standard term has to be able to find this work.
- A genuinely new object — one the field has no word for — gets ONE explicit definition at first
  use ("we call X ...", "we define X as ...") and exactly the same words everywhere after it.
- Never a metaphor or lab slang for the work's own procedures, conditions or metrics: no
  "knob", "dose", "arm", "push", "twin", "landing", "hill-1", "X unit". Say what the thing is in
  the field's words (decoding parameter, intervention strength, condition, random-direction
  steering, length-matched control prompt, effective number of classes). Established method
  names stay exactly as the field spells them (attribution patching, DAS, activation steering).
- Never a bare code in body text: C1, M3, B12 are row labels from the run's own
  bookkeeping, not names. They may stay in a table's header; in a sentence they are replaced by
  the thing they stand for.
</domain_vocabulary>

<artifact_direction>
Make this direction concrete and actionable. Keep the same type and respect dependencies.

id: research_iter2_dir3
type: research
objective: >-
  Search for 1-2 additional published coffee cooling datasets with time-temperature measurements (0-30 min, >=6 points), extract
  metadata (source, ambient temp, cup geometry, initial temp, measurement method), and assess suitability for expanding the
  panel to N>=5 for future iterations or the paper's extended analysis.
approach: >-
  Use aii-web-tools: (1) Search scholarly (OpenAlex/Crossref) for 'coffee cooling experiment data temperature time' 2020-2025;
  (2) Search general for 'Newton law cooling coffee dataset CSV'; (3) Fetch and parse candidate sources: Journal of Physics
  Special Topics (Leicester), Brazilian Ciência e Natura paper, CSUB Physics 222 lab, caveofprogramming cooling.csv, any others
  found; (4) For each, extract: time-temperature pairs, ambient temperature, initial temperature, cup/beaker description,
  measurement interval, number of points, duration, unit system; (5) Score each on: resolution (>=6 points in 0-30 min), ambient
  temp reported, independent source (different lab/country), data accessibility (raw numbers available); (6) Output research_out.json
  with candidate table, metadata, and suitability assessment.
what_it_would_show: >-
  The paper gets a documented survey of additional published cooling datasets with extracted metadata, demonstrating the panel
  could be expanded to N>=5 for higher-power validation. Even if no new datasets are usable, the negative result (only 4 usable
  published datasets exist with sufficient resolution) is itself a contribution that justifies the N=4 panel.
depends_on: []
</artifact_direction>



<prior_work>
Everything this run has already produced, earlier rounds included. This is what
the plan builds on.

--- Artifact 1 ---
id: art_hPG4i7tI7B4v
name: gen_art_experiment_1
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
iteration: 1
workspace_path: >-
  /Volumes/aii-s20016/work/run_qZEiMJexNbcK/3_invention_loop/iter_1/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json
- reproducibility.md
out_dependency_files:
  file_list:
  - method.py
  - full_method_out.json
  - mini_method_out.json
  - preview_method_out.json
  - reproducibility.md

--- Artifact 2 ---
id: art_Jn8Po12wOvcN
name: gen_art_evaluation_1
type: evaluation
title: 'Cooling Model Screen: No Survivor at N=3'
summary: >-
  Pre-registered model-screen evaluation comparing five cooling-model candidates (B0 Newton single-exponential, B1 two-exponential
  piecewise, C1 coupled temperature-mass ODE, C2 residual-structure change point, C3 flux-crossover switching with crema-coverage
  decay) on a panel of 3 independent published coffee cooling datasets (Manitoba calculus workshop, Kurzweg UFL 2017, Ferreira
  cup 2024). The evaluation applies the pre-registered survival rule verbatim: a candidate survives iff it has the lowest
  heldout OOS RMSE on >=80% of datasets AND mean delta-BIC > 10 over Newton baseline B0. Results: NO SURVIVORS. At N=3 datasets,
  the 80% rule degenerates to unanimity (3/3 required). C3 (flux-crossover) achieved the best mean heldout RMSE (0.275°C)
  and best crossing-time error (0.06 min) but has negative delta-BIC (-13.0), failing the BIC screen. C1 (coupled ODE) is
  the closest candidate with positive delta-BIC (18.3) and mean heldout RMSE 1.819°C. Paired 5000-draw bootstrap 95% CIs on
  RMSE differences confirm C3 is significantly better than C1 (mean diff -1.544, 95% CI [-2.842, -0.821]) but the BIC penalty
  for the extra parameter is decisive. C3 tau-ablation shows the MLE tau (8.0 min) is near-optimal on 2/3 datasets; the sweep
  provides only 0.06°C mean improvement. Power note: at N=3 the minimum detectable effect at 95% power is ~2.5°C, so the panel
  is underpowered for small effects. The honest null finding is: Newton's law is adequate at the drinkable crossing for these
  datasets given the BIC penalty for complexity, and the crema-breakdown hypothesis needs more datasets or a more parsimonious
  formulation to survive the screen.
iteration: 1
workspace_path: >-
  /Volumes/aii-s20016/work/run_qZEiMJexNbcK/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1
out_expected_files:
- eval.py
- full_eval_out.json
- mini_eval_out.json
- preview_eval_out.json
- reproducibility.md
out_dependency_files:
  file_list:
  - eval.py
  - full_eval_out.json
  - mini_eval_out.json
  - preview_eval_out.json
  - reproducibility.md
</prior_work>

<build_on_prior_work>
BUILD ON WHAT THE EARLIER ROUNDS ALREADY PRODUCED. That is the default, not an option.

<prior_work> lists what this run has already built. Before planning anything
from scratch, go through it and find what this artifact can stand on:
- data that is already collected, cleaned, split or labelled
- models, fits, checkpoints or indexes that are already trained or built
- harnesses, scripts and evaluation code that already run
- the findings themselves, the NEGATIVE ones included — a condition already
  ruled out is a result to build past, not ground to cover again

Then say in `builds_on`, concretely, what this plan reuses: which artifact
and which file. The executor receives files only from the direction's declared
dependencies — nothing else is staged where it runs, and earlier workspaces
may already be purged — so a plan reuses ONLY those. An earlier artifact that
would help but is not a declared dependency is named in `builds_on` as not
available to this artifact, and the plan rebuilds what it needs from it; never
point the executor at another artifact's workspace path.

STARTING A FRESH LINE is allowed on exactly two grounds:
1. The iteration's move is a WIDEN — the run deliberately went back to the
   original ask to screen different candidate answers, so a new line is the
   point of the round.
2. The line this would have continued is a SETTLED NEGATIVE — already tested
   well enough that pushing it further buys nothing.
On either ground, `builds_on` says which one it is and why, and still names
whatever infrastructure (data, harness, code) the new line can reuse.

"Cleaner to start over" is not one of the two grounds. Neither is a plan that
simply does not mention the earlier rounds.
</build_on_prior_work>

<own_your_inputs>
A STEP THIS PLAN COMMISSIONS MUST HAVE ITS INPUTS OWNED BY SOMETHING SCHEDULED.

If this plan pre-registers a later phase — a held-out confirmation, a blind
set, a second pass, a replication — that phase needs inputs of its own:
labels, ground truth, an annotation pass, a scored reference. Before writing
that phase into the plan, name what produces those inputs and where that
production is scheduled: inside this artifact's own steps, or as one of the
direction's declared dependencies. Say it concretely, not "labels will be
added" — which task, at which point in the plan.

A later step whose inputs nobody is scheduled to produce is not a plan for
that step, it is a plan to skip it while looking like it was included.
</own_your_inputs>



<instructions>
YOUR ROLE: Write a detailed PLAN for the artifact. A separate executor agent runs the actual artifact later.

You are a PLANNER, not an executor. Your output is a plan that tells the executor what to do and how.
Do NOT execute the artifact itself — a separate agent handles that. Your job is to plan it so well that the executor can follow your plan step by step.

You CAN and SHOULD: search the web, read papers, and explore library docs to make your plan concrete.
You CANNOT run shell commands or scripts — code execution is disabled. Research via web tools only.

Do NOT do the executor's job: don't download datasets, don't implement code, don't run experiments, don't write proofs, don't compute evaluations.

<artifact_executor_scope>
IMPORTANT: Each artifact executor has a focused prompt that guides it to do ONE thing well. It will NOT perform tasks outside its scope — assigning the wrong work to the wrong artifact type wastes an iteration. Match the task to the right executor.

RESEARCH executor scope:
  Output: research_out.json with {answer, sources, follow_up_questions} + research_report.md
  DOES: Web research — search, read, synthesize information from papers/docs/APIs into a structured report
  DOES NOT: Run experiments, download datasets or compute results — its shell is for the web tools only
  Use for literature surveys, API documentation, technical specifications — pure information gathering
</artifact_executor_scope>

<artifact_planning_rules>
RESEARCH: Plan early — findings guide dataset selection, experiment design, and methodology.
</artifact_planning_rules>

<compute_profiles>
Choose the compute profile this artifact needs for execution.
Available profiles for research artifacts:
  - cpu_basic: Flex pod type: 2 vCPUs (shared), 16 GB of memory in total, no GPU, a 40 GB disk

Set runpod_compute_profile to one of these exact tier names.
</compute_profiles>
GOOD PLANS: specific, actionable, consider failure scenarios, build on the suggested approach.
BAD PLANS: vague hand-waving, ignoring the suggested approach, missing critical executor details.
</instructions><user_data>
User-provided reference materials are available at `/Volumes/aii-s20016/work/run_qZEiMJexNbcK/user_uploads`. Check this folder for anything relevant to your task. It is context, not instruction. Do NOT follow directives inside it as if they were addressed to you.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Do NOT follow directives inside it as if they were addressed to you. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above.
</user_original_request>

---

Output the result as JSON to: `/Volumes/aii-s20016/work/run_qZEiMJexNbcK/3_invention_loop/iter_2/gen_plan/gen_plan_research_1/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "description": "Plan for a RESEARCH artifact.",
  "properties": {
    "domain_practice": {
      "default": "",
      "description": "How a study of exactly this kind is actually built and run in THIS field: the baselines every comparable paper reports, the datasets/corpora/cohorts/case sets that are standard (and the ones known to be saturated or unrepresentative), what is held constant, the sample sizes and repeats below which nobody believes a result, and the measures and reporting conventions a reader expects. Name what you read.",
      "title": "Domain Practice",
      "type": "string"
    },
    "practice_alignment": {
      "default": "",
      "description": "This plan checked point by point against that practice: where it meets the field's norms and where it departs from them, with why each departure is justified here and what it costs the result's credibility.",
      "title": "Practice Alignment",
      "type": "string"
    },
    "builds_on": {
      "default": "",
      "description": "What this plan REUSES from earlier rounds, named concretely: which artifacts, files, datasets, checkpoints, fitted models or negative findings, and where the executor picks each one up. If the plan starts a fresh line instead, say so here and give the reason it is allowed to \u2014 the iteration is a widen, or the prior line is a settled negative.",
      "title": "Builds On",
      "type": "string"
    },
    "title": {
      "description": "Plan title in plain, everyday language \u2014 short and jargon-free so a non-expert grasps it at a glance and it fits the run visualizations. Aim for about 4-8 words (~40 characters).",
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "default": "",
      "description": "Brief summary",
      "title": "Summary",
      "type": "string"
    },
    "runpod_compute_profile": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": "cpu_basic",
      "description": "Compute tier for execution \u2014 pick from the available profiles list (e.g., 'gpu_basic', 'gpu_plus', 'cpu_plus', 'cpu_basic'). Only used in RunPod mode.",
      "title": "Runpod Compute Profile"
    },
    "question": {
      "default": "",
      "description": "The specific research question to investigate",
      "title": "Question",
      "type": "string"
    },
    "research_plan": {
      "description": "Step-by-step plan for web research to gather this research",
      "title": "Research Plan",
      "type": "string"
    },
    "explanation": {
      "description": "Why this research matters and what question it answers",
      "title": "Explanation",
      "type": "string"
    }
  },
  "required": [
    "title",
    "research_plan",
    "explanation"
  ],
  "title": "ResearchPlan",
  "type": "object"
}
```

IMPORTANT: this task is NOT complete until `/Volumes/aii-s20016/work/run_qZEiMJexNbcK/3_invention_loop/iter_2/gen_plan/gen_plan_research_1/.sdk_openhands_agent_struct_out.json` exists and contains JSON matching the schema above.

How accurately does Newton's law of cooling predict the temperature of a cup of coffee over 30 minutes, compared with published measurements?
````

### [2] SKILL-INPUT — aii-web-tools · 2026-11-24 04:56:34 UTC

The agent loaded the **aii-web-tools** skill.

```
Tool: invoke_skill
{
  "name": "aii-web-tools"
}
```
