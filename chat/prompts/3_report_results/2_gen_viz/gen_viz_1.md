# gen_viz_1 — report_results

> Phase: `gen_paper_repo` · `gen_viz`
> Run: `run_qZEiMJexNbcK` — Newton's Law of Cooling Deviates by 6°C After 30 Minutes of Coffee Cooling
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_viz_1` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-11-24 13:26:17 UTC

````
<task>
Generate a publication-quality CONCEPT figure for a top-tier venue research paper that follows the provided specification.

Use the aii-concept-fig-gen skill to generate the figure in the aspect ratio from the spec. ALWAYS pass `--model sunburst --style concept` to EVERY concept_fig_gen.py call (this run uses the **sunburst** image tier).

ONLY THIS ROUTE. A drawn figure — a flowchart, an overview, a pipeline or method diagram, the flagship — is made by concept_fig_gen.py on the `sunburst` tier and nothing else. Never pass another `--model`, never call another image model or image API, and never draw it yourself: no matplotlib, graphviz, TikZ, SVG, HTML or PIL drawing. PIL is for the margin crop below and nothing more. `--style concept` appends the paper's concept style — lettered panels with bold titles, the field's own objects, meaningful saturated colour, large labels, the paper's typeface, its palette, a canvas filled edge to edge — so the tool carries it on every call instead of you having to remember it in every prompt.

CONCEPT RULES. The figure is printed 6.5 inches wide on a single-column page, and it looks like the Figure 1 of a top recent paper in this paper's field: concrete, rich and readable, never a sparse cartoon.
- The canvas is 16:9, and a concept figure is never wider than 2:1.
- 2 to 4 lettered panels, (a), (b), (c), read LEFT TO RIGHT in the order the specification gives, each with a short bold title that says what the panel shows.
- About 80 to 120 words in the whole figure, most of them the paper's own text: a real prompt or input, the decisive tokens, a one-line code snippet, the real method and model names. Labels are short role names or sentence fragments of at most about eight words; no paragraphs.
- Real numbers: at least one (in Figure 1, the headline number) and the few others the specification gives — a k, a temperature, a count, a share — each exactly as written there. Short plain notation (pass@k, T = 0.8) and one-line code or token snippets are welcome; display equations and derivations stay in the paper's text.
- Draw the field's own objects: token strips, code lines, layer stacks, probability bars, sampling trees, chat bubbles, a small result thumbnail. Never a generic icon (a gear, a brain, a light bulb, a thermometer, a question mark) and never a SLOGAN PANEL, a title over an icon, a lone word or an empty box.
- LARGE labels, the panel titles largest, every word readable at 6.5 inches wide. Never ask for small text, fine print, thin lines or generous white space: the figure fills its canvas.
- NO OVERALL FIGURE TITLE in the image: the caption is the figure's title. The `Title` in the specification names the figure for the paper and is never drawn; only the panels carry titles.
- COLOUR CARRIES MEANING: 2 to 4 saturated colours, each with one meaning that stays the same in every panel and that the caption can name; grey for what is shared or de-emphasised; a white background, thin grey rounded panel frames, and pastel only as a light tint behind text. ONE COLOUR MAP FOR THE WHOLE PAPER, the same one its data figures use, so a colour means the same thing in every figure. Where the figure contrasts the paper's own method with a baseline, colour each element by its ROLE:
  - `#0173B2` blue: the paper's own method, model or condition
  - `#949494` grey: the main baseline, control or reference
  - `#DE8F05` amber: the first further comparator
  - `#029E73` green: the second
  - `#CC78BC` violet: the third
  Where it has no such contrast, colour by STAGE or CONCEPT instead, one of the same colours per stage, outcome or branch, keeping any colour a data figure of the paper already gives the same thing.

DOMAIN EXEMPLARS — before your first generation, look at how this paper's field draws its Figure 1s (about 6 to 10 tool calls and a few minutes, no more):
1. Take the arXiv ids after "Style references:" in the specification. With fewer than two, run ONE web search for "<2 to 4 keywords of this paper's field> arXiv 2025 ICLR OR NeurIPS OR ICML" and add the ids of the most-cited recent papers that name a method or phenomenon.
2. Fetch their Figure 1 pages in one call, which returns on its own at its deadline: `$PY "$SKILL_DIR/scripts/fig1_exemplars.py" <id> <id> <id> --out exemplars`
3. Read each page it rendered. Keep the two or three whose Figure 1 is an overview (a pipeline, a worked example, a concept beside a result); for a pure results chart, fetch the next candidate rather than lowering the bar. If none can be fetched, go on with these rules alone.
4. Write `fig1_style_note.md`: 3 to 5 lines per kept exemplar (its layout, roughly how many words, the real objects and numbers it shows, what its colours mean), then which one this figure adapts and how, then the claim this figure must show and its draft caption.
5. Write your image prompt as that adaptation, drawn with this paper's own objects and numbers. Never copy an exemplar's content. You may add domain detail the specification, caption and summary support (a real token, a code line, a model name), never a result: no number they do not state.

Be detailed about LAYOUT and exact about TEXT in your image generation prompt: say where each panel, object and arrow sits, its shape and its colour, and give the exact text, tokens and numbers each one shows, copied from the specification. Then END the prompt with a separate sentence listing the words that must appear, verbatim — "The boxes read Tokenizer, Transformer, Classifier." Naming them inside the layout sentence instead is what turns Encoder into `Enc:der`; every measured run that stated them as their own closing sentence spelled all of them correctly, and word length made no difference either way.

Create a subfolder `fig1_all/` in your workspace for ALL attempts.

ONE PHASE — explore layouts (HARD LIMIT: 7 attempts). `sunburst` ignores `--image-size` and always renders at its own maximum size, so an edit-mode "upscale" of a chosen draft returns the same picture for the price of a new one. There is NO upscale phase on this tier: spend those generations on layouts instead.
- Generate at `--model sunburst --image-size 1K`. Save attempts as `fig1_all/fig1_v0_it1.jpg`, `fig1_all/fig1_v0_it2.jpg`, … up to `_it7.jpg`.
- Keep the specification's reading order, its panels left to right, in every attempt. Make your first two or three attempts genuinely DIFFERENT LAYOUTS within it — the panels' widths, where the worked example sits, how the panels are joined — not rewordings of one layout. Then refine the clearest layout with the attempts left.
- After EACH attempt, read the image back and verify it against the checklist below. Copy every attempt that passes it to the deliverable, crop it and run the print check (DELIVERABLE below). Stop only once one passes the checklist AND the print check, its style verdict included: a clean-looking render the style check fails is not done. Otherwise keep going until the 7 attempts are spent, and deliver the best. Every paid generation draws on this run's OpenRouter budget for Report results: $7 USD for the ENTIRE phase that writes the paper and repo, not just this figure. Every other agent and step in that phase spends from that SAME shared pot, so use fewer or cheaper attempts when you are unsure. $7 USD of it was held back for concept figures, and the pot is enforced by AI Inventor: a generation past it is refused (HTTP 403, 'AI Inventor per-run OpenRouter budget reached'), and retrying will not help.
- The chosen attempt is the final figure.

DELIVERABLE:
- Copy the chosen final image to your workspace root as: fig1_v0.jpg — the chosen attempt.
- Crop its blank margins in code before you deliver it: with PIL, take the bounding box of
  pixels darker than about 245, add a 20-pixel margin, and save it over fig1_v0.jpg as
  JPEG quality 95. Read the cropped file back. Never leave cropping to LaTeX options: the paper,
  the repo and the website must all show this one file.
- RUN THE PRINT CHECK on the cropped file, every time you crop, with your style note and the
  exemplar pages you kept:
  `$PY "$SKILL_DIR/scripts/concept_print_check.py" fig1_v0.jpg --aspect 16:9 --style-note fig1_style_note.md --reference exemplars/<id>/fig1_page.png`
  (`$PY` and `$SKILL_DIR` as the skill's Setup section sets them; one `--reference` per kept
  page). It re-checks the aspect after the crop — a crop more than 15% wider than 16:9
  means the drawing did not fill the canvas height — downscales the figure to 650 px wide, the
  size the page prints it at, and has a vision model list every label it cannot read there. Then
  it judges MEANING AND STYLE against the exemplars: does each panel show its point with the
  field's own objects and real numbers, and is the figure about as rich as they are? Exit 1 is a
  FAIL: read its reasons and regenerate (it counts as an attempt). An unreadable label or a strip
  wants shorter labels and larger text, filling the full height of the canvas; a style failure
  wants what its problems name. Exit 2 means it could not reach its vision model (or would bill on
  a $0 run): the aspect was still checked, so deliver. Read back the
  `fig1_v0_print650.png` it writes: that is what a reader sees.
- Leave `fig1_v0.print_check.json` beside the figure. The step reads it, and it vouches
  only for the exact file it checked. A figure delivered without a passing check, or checked
  without `--style-note`, gets ONE retry, and a figure still wider than 2.6:1 after its crop is
  refused outright.
- The file `fig1_v0.jpg` is the deliverable — everything in `fig1_all/` is reference only.

Verification checklist (apply after EVERY generation). Check for:
- MISSPELLED labels. Read every word in the image letter by letter against the word you asked for. This is the most common defect by a wide margin — `erooder` for Encoder, `routter` for Router, `conveged?` for converged? — and it is the one that survives a glance, because the shape of the word is right
- Invented text you never asked for. A prompt ending "no text of any kind" came back lettered with `Kat q` and fake axis ticks, so absence has to be checked too, not assumed
- A box, arrow or panel that is duplicated, missing, or pointing nowhere, even when every word in the image is spelled correctly
- Text too small to read at 6.5 inches wide, or a label over about eight words
- An overall title or headline drawn above the panels: the figure carries none, only its panel titles
- A SPARSE CARTOON: a slogan panel (a title over a generic icon, a lone word or an empty box), no real number, no worked example, or a figure far sparser than the exemplars you kept
- A panel whose drawing does not show the claim its title makes, or panels out of the left-to-right order the specification gives
- A display equation or derivation, or a code snippet longer than a line or two
- Empty bands or wide margins the drawing does not fill, or a strip much wider than 16:9
- Overlapping, touching, cut-off or blurry labels, and poor contrast
- Font: serif labels in the style of LaTeX's Computer Modern, the paper's typeface, which the style preset asks for, to match the paper's other figures. Do not spend an attempt on the typeface alone when everything else is right

If ANY issue is found — even minor — do another attempt (within the 7-attempt limit). Do NOT accept a figure with problems as the final figure.

Change the prompt only when the prompt is what was wrong — a word you never specified, an element you forgot to name. For a defect the prompt already rules out, re-run it UNCHANGED: the same prompt sent twice gave a correct three-box chain once and four boxes with one label repeated the other time. Rewriting a prompt that was already right spends one of your 5 attempts on a variable that was not the cause.
</task>

<figure_specification>
Figure ID: fig1
Title: How five models represent coffee cooling
Caption: Five models for coffee cooling: (a) Newton's single-exponential law assumes constant cooling rate; (b) multi-regime models allow a transition between fast and slow cooling; (c) the Bayesian change-point detects the transition from residual structure. Blue marks multi-regime models, grey marks Newton's law.
Image Generation Description: Three panels left to right, each in a thin grey rounded frame on white, joined by thick dark arrows, filling the whole canvas. Panel (a) titled '(a) Newton\'s law': a cooling curve plot with x-axis 'Time (min) 0–30' and y-axis 'Temperature (°C) 20–95', showing a smooth blue exponential curve from 90°C to 40°C, and grey dots scattered around it with increasing deviation after 15 min; a label 'single-exponential: 2 parameters' and 'error peaks at 6.0°C at 30 min' in bold. Panel (b) titled '(b) Regime switching': the same axes, showing two curves — a blue piecewise curve with a visible kink at ~12 min labelled 'flux-crossover: \u03c4 = 5.5 min' and a green curve with a kink at ~10 min labelled 'piecewise: free transition' — both tracking the grey dots more closely; a label 'two regimes: fast then slow'. Panel (c) titled '(c) Change-point detection': the same axes, showing a red curve that adapts to the data with a kink at ~11 min labelled 'Bayesian change-point: detected from residuals' and 'RMSE 1.66°C vs 4.04°C'; a small inset showing the residual plot with a clear sign change at the transition point. Blue for multi-regime models, grey for Newton, red for Bayesian change-point in every panel. Style references: arXiv 2401.01234, 2305.06162.
Aspect Ratio: 16:9
Summary: Overview of the five cooling models and their representation of the regime transition
</figure_specification>

<critical_requirements>
1. Draw what the specification describes, adding only domain detail the specification, caption and summary support (a real token, a code line, a model name). Invent no result: no number they do not state, and no stage the method does not have
2. The concept rules above are hard rules: 2 to 4 lettered panels read left to right, no overall figure title, about 80 to 120 words with at least one real number, the field's own objects and no slogan panels, no display equations, a canvas no wider than 2:1
3. FONT: serif labels in the style of LaTeX's Computer Modern, the paper's typeface — `--style concept` asks for it on every call
4. Publication camera-ready style, in the manner of the exemplars you kept: white background, thin grey rounded panel frames, saturated colour that means something, no 3D effects/shadows/gradients. Follow aii-concept-fig-gen skill for image generation, prompting best practices, and figure type templates
5. TEXT SIZE: LARGE labels that never overlap or touch. Give a crowded figure room by shortening labels or dropping a secondary one, never by dropping the worked example or the real numbers, never by going wider and never by asking for smaller text
6. RESOLUTION: Explore layouts at 1K in the one phase above; no upscale or edit passes on this tier. Keep the aspect ratio from the spec. The paper's Figure 1 overview stays 16:9, never wider than 2:1
7. MANDATORY VERIFICATION: After EVERY generation attempt, read the image and check spelling first, then layout and meaning, then the print check (legibility and style) on the cropped deliverable. If anything is wrong, regenerate. Do NOT stop at the first clean-looking attempt
</critical_requirements>


<comparison_completeness>
If this figure's title, caption or summary names specific checkpoints, models or
variants being COMPARED — "ours vs baseline", "the base and the abliterated model",
"across the three checkpoints" — every one of them named there MUST appear in the
rendered figure as its own bar, curve, point or panel. Before you render, list every
comparator the specification names and check each one off as you draw it. A
comparison figure that quietly drops one of its own named comparators is wrong even
when every bar it does draw is numerically correct — the missing one is invisible to
anyone who was not told to look for it, which is what makes it worse than an
obviously incomplete figure.
</comparison_completeness>


<caption_from_the_rendered_figure>
The caption in <figure_specification> is a DRAFT, written before this figure existed by a step
that never saw it. After your final render, read the final image back and write the figure's
caption into the `caption` field of your output. It replaces the draft caption everywhere this
figure appears: the paper, the report and the paper's website.
- AT MOST 60 WORDS. One sentence on what the figure shows and what it argues, then only
  what a reader cannot see for themselves: what a colour or marker means where no legend says it,
  what the error bars are, the sample size. Do not re-list the axes, panels or labels the image
  already names; the detail belongs in the paper's text. A 300-word caption that walks the reader
  through every panel is the failure this limit exists to stop.
- Name a colour, marker, panel or series only if it is in the image. Where the draft caption and
  the image disagree (colours said to encode models when they encode languages, an axis the panel
  does not plot, a grey series that was never drawn), the image wins.
- Keep what is still true of the draft: the data, the sample size, what the error bars are, and
  the takeaway. Keep it LaTeX-ready in the same form as the draft caption.
- State no number the figure and its data do not carry.
- A figure with no data behind it that could pass for data (a drawn network, a map, a snapshot)
  is called a schematic illustration in its caption.
</caption_from_the_rendered_figure>


---

Output the result as JSON to: `/Volumes/aii-s20003/work/run_qZEiMJexNbcK/4_gen_paper_repo/_3_gen_viz/gen_viz_1/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "VizExpectedFiles": {
      "description": "Expected output files from viz generation.",
      "properties": {
        "image_path": {
          "description": "Path to the generated figure image file. Example: 'fig1_v0.jpg'",
          "title": "Image Path",
          "type": "string"
        }
      },
      "required": [
        "image_path"
      ],
      "title": "VizExpectedFiles",
      "type": "object"
    }
  },
  "description": "Structured output from viz figure generation agent.",
  "properties": {
    "title": {
      "description": "Figure title in plain, everyday language \u2014 short and jargon-free so a non-expert grasps it at a glance. Aim for about 4-8 words (~40 characters).",
      "maxLength": 90,
      "minLength": 12,
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "description": "Brief summary of the generated figure: what it shows, style, any issues fixed",
      "maxLength": 5000,
      "minLength": 500,
      "title": "Summary",
      "type": "string"
    },
    "caption": {
      "description": "The figure's caption, written from the FINAL rendered image after you read it back, in the same LaTeX-ready form as the draft caption. AT MOST ABOUT 60 WORDS: one sentence on what the figure shows and argues, then only what a reader cannot see for themselves (an encoding no legend states, what the error bars are, the sample size). Do not re-list the axes, panels or labels the image already names. It replaces the draft caption in the paper, the report and the paper's website. Name only axes, colours, markers, panels and series that are in the image.",
      "maxLength": 480,
      "minLength": 20,
      "title": "Caption",
      "type": "string"
    },
    "out_expected_files": {
      "$ref": "#/$defs/VizExpectedFiles",
      "description": "Output file you created. Must include the generated figure image path."
    }
  },
  "required": [
    "title",
    "summary",
    "caption",
    "out_expected_files"
  ],
  "title": "VizFigureOutput",
  "type": "object"
}
```

IMPORTANT: this task is NOT complete until `/Volumes/aii-s20003/work/run_qZEiMJexNbcK/4_gen_paper_repo/_3_gen_viz/gen_viz_1/.sdk_openhands_agent_struct_out.json` exists and contains JSON matching the schema above.

How accurately does Newton's law of cooling predict the temperature of a cup of coffee over 30 minutes, compared with published measurements?
````

### [2] SKILL-INPUT — aii-concept-fig-gen · 2026-11-24 13:26:33 UTC

The agent loaded the **aii-concept-fig-gen** skill.

```
Tool: invoke_skill
{
  "name": "aii-concept-fig-gen"
}
```

### [3] SKILL-INPUT — aii-web-tools · 2026-11-24 13:29:28 UTC

The agent loaded the **aii-web-tools** skill.

```
Tool: invoke_skill
{
  "name": "aii-web-tools"
}
```

### [4] SYSTEM-USER prompt · 2026-11-24 21:07:01 UTC

````
<task>
Generate a publication-quality CONCEPT figure for a top-tier venue research paper that follows the provided specification.

Use the aii-concept-fig-gen skill to generate the figure in the aspect ratio from the spec. ALWAYS pass `--model sunburst --style concept` to EVERY concept_fig_gen.py call (this run uses the **sunburst** image tier).

ONLY THIS ROUTE. A drawn figure — a flowchart, an overview, a pipeline or method diagram, the flagship — is made by concept_fig_gen.py on the `sunburst` tier and nothing else. Never pass another `--model`, never call another image model or image API, and never draw it yourself: no matplotlib, graphviz, TikZ, SVG, HTML or PIL drawing. PIL is for the margin crop below and nothing more. `--style concept` appends the paper's concept style — lettered panels with bold titles, the field's own objects, meaningful saturated colour, large labels, the paper's typeface, its palette, a canvas filled edge to edge — so the tool carries it on every call instead of you having to remember it in every prompt.

CONCEPT RULES. The figure is printed 6.5 inches wide on a single-column page, and it looks like the Figure 1 of a top recent paper in this paper's field: concrete, rich and readable, never a sparse cartoon.
- The canvas is 16:9, and a concept figure is never wider than 2:1.
- 2 to 4 lettered panels, (a), (b), (c), read LEFT TO RIGHT in the order the specification gives, each with a short bold title that says what the panel shows.
- About 80 to 120 words in the whole figure, most of them the paper's own text: a real prompt or input, the decisive tokens, a one-line code snippet, the real method and model names. Labels are short role names or sentence fragments of at most about eight words; no paragraphs.
- Real numbers: at least one (in Figure 1, the headline number) and the few others the specification gives — a k, a temperature, a count, a share — each exactly as written there. Short plain notation (pass@k, T = 0.8) and one-line code or token snippets are welcome; display equations and derivations stay in the paper's text.
- Draw the field's own objects: token strips, code lines, layer stacks, probability bars, sampling trees, chat bubbles, a small result thumbnail. Never a generic icon (a gear, a brain, a light bulb, a thermometer, a question mark) and never a SLOGAN PANEL, a title over an icon, a lone word or an empty box.
- LARGE labels, the panel titles largest, every word readable at 6.5 inches wide. Never ask for small text, fine print, thin lines or generous white space: the figure fills its canvas.
- NO OVERALL FIGURE TITLE in the image: the caption is the figure's title. The `Title` in the specification names the figure for the paper and is never drawn; only the panels carry titles.
- COLOUR CARRIES MEANING: 2 to 4 saturated colours, each with one meaning that stays the same in every panel and that the caption can name; grey for what is shared or de-emphasised; a white background, thin grey rounded panel frames, and pastel only as a light tint behind text. ONE COLOUR MAP FOR THE WHOLE PAPER, the same one its data figures use, so a colour means the same thing in every figure. Where the figure contrasts the paper's own method with a baseline, colour each element by its ROLE:
  - `#0173B2` blue: the paper's own method, model or condition
  - `#949494` grey: the main baseline, control or reference
  - `#DE8F05` amber: the first further comparator
  - `#029E73` green: the second
  - `#CC78BC` violet: the third
  Where it has no such contrast, colour by STAGE or CONCEPT instead, one of the same colours per stage, outcome or branch, keeping any colour a data figure of the paper already gives the same thing.

DOMAIN EXEMPLARS — before your first generation, look at how this paper's field draws its Figure 1s (about 6 to 10 tool calls and a few minutes, no more):
1. Take the arXiv ids after "Style references:" in the specification. With fewer than two, run ONE web search for "<2 to 4 keywords of this paper's field> arXiv 2025 ICLR OR NeurIPS OR ICML" and add the ids of the most-cited recent papers that name a method or phenomenon.
2. Fetch their Figure 1 pages in one call, which returns on its own at its deadline: `$PY "$SKILL_DIR/scripts/fig1_exemplars.py" <id> <id> <id> --out exemplars`
3. Read each page it rendered. Keep the two or three whose Figure 1 is an overview (a pipeline, a worked example, a concept beside a result); for a pure results chart, fetch the next candidate rather than lowering the bar. If none can be fetched, go on with these rules alone.
4. Write `fig1_style_note.md`: 3 to 5 lines per kept exemplar (its layout, roughly how many words, the real objects and numbers it shows, what its colours mean), then which one this figure adapts and how, then the claim this figure must show and its draft caption.
5. Write your image prompt as that adaptation, drawn with this paper's own objects and numbers. Never copy an exemplar's content. You may add domain detail the specification, caption and summary support (a real token, a code line, a model name), never a result: no number they do not state.

Be detailed about LAYOUT and exact about TEXT in your image generation prompt: say where each panel, object and arrow sits, its shape and its colour, and give the exact text, tokens and numbers each one shows, copied from the specification. Then END the prompt with a separate sentence listing the words that must appear, verbatim — "The boxes read Tokenizer, Transformer, Classifier." Naming them inside the layout sentence instead is what turns Encoder into `Enc:der`; every measured run that stated them as their own closing sentence spelled all of them correctly, and word length made no difference either way.

Create a subfolder `fig1_all/` in your workspace for ALL attempts.

ONE PHASE — explore layouts (HARD LIMIT: 7 attempts). `sunburst` ignores `--image-size` and always renders at its own maximum size, so an edit-mode "upscale" of a chosen draft returns the same picture for the price of a new one. There is NO upscale phase on this tier: spend those generations on layouts instead.
- Generate at `--model sunburst --image-size 1K`. Save attempts as `fig1_all/fig1_v0_it1.jpg`, `fig1_all/fig1_v0_it2.jpg`, … up to `_it7.jpg`.
- Keep the specification's reading order, its panels left to right, in every attempt. Make your first two or three attempts genuinely DIFFERENT LAYOUTS within it — the panels' widths, where the worked example sits, how the panels are joined — not rewordings of one layout. Then refine the clearest layout with the attempts left.
- After EACH attempt, read the image back and verify it against the checklist below. Copy every attempt that passes it to the deliverable, crop it and run the print check (DELIVERABLE below). Stop only once one passes the checklist AND the print check, its style verdict included: a clean-looking render the style check fails is not done. Otherwise keep going until the 7 attempts are spent, and deliver the best. Every paid generation draws on this run's OpenRouter budget for Report results: $7 USD for the ENTIRE phase that writes the paper and repo, not just this figure. Every other agent and step in that phase spends from that SAME shared pot, so use fewer or cheaper attempts when you are unsure. $7 USD of it was held back for concept figures, and the pot is enforced by AI Inventor: a generation past it is refused (HTTP 403, 'AI Inventor per-run OpenRouter budget reached'), and retrying will not help.
- The chosen attempt is the final figure.

DELIVERABLE:
- Copy the chosen final image to your workspace root as: fig1_v0.jpg — the chosen attempt.
- Crop its blank margins in code before you deliver it: with PIL, take the bounding box of
  pixels darker than about 245, add a 20-pixel margin, and save it over fig1_v0.jpg as
  JPEG quality 95. Read the cropped file back. Never leave cropping to LaTeX options: the paper,
  the repo and the website must all show this one file.
- RUN THE PRINT CHECK on the cropped file, every time you crop, with your style note and the
  exemplar pages you kept:
  `$PY "$SKILL_DIR/scripts/concept_print_check.py" fig1_v0.jpg --aspect 16:9 --style-note fig1_style_note.md --reference exemplars/<id>/fig1_page.png`
  (`$PY` and `$SKILL_DIR` as the skill's Setup section sets them; one `--reference` per kept
  page). It re-checks the aspect after the crop — a crop more than 15% wider than 16:9
  means the drawing did not fill the canvas height — downscales the figure to 650 px wide, the
  size the page prints it at, and has a vision model list every label it cannot read there. Then
  it judges MEANING AND STYLE against the exemplars: does each panel show its point with the
  field's own objects and real numbers, and is the figure about as rich as they are? Exit 1 is a
  FAIL: read its reasons and regenerate (it counts as an attempt). An unreadable label or a strip
  wants shorter labels and larger text, filling the full height of the canvas; a style failure
  wants what its problems name. Exit 2 means it could not reach its vision model (or would bill on
  a $0 run): the aspect was still checked, so deliver. Read back the
  `fig1_v0_print650.png` it writes: that is what a reader sees.
- Leave `fig1_v0.print_check.json` beside the figure. The step reads it, and it vouches
  only for the exact file it checked. A figure delivered without a passing check, or checked
  without `--style-note`, gets ONE retry, and a figure still wider than 2.6:1 after its crop is
  refused outright.
- The file `fig1_v0.jpg` is the deliverable — everything in `fig1_all/` is reference only.

Verification checklist (apply after EVERY generation). Check for:
- MISSPELLED labels. Read every word in the image letter by letter against the word you asked for. This is the most common defect by a wide margin — `erooder` for Encoder, `routter` for Router, `conveged?` for converged? — and it is the one that survives a glance, because the shape of the word is right
- Invented text you never asked for. A prompt ending "no text of any kind" came back lettered with `Kat q` and fake axis ticks, so absence has to be checked too, not assumed
- A box, arrow or panel that is duplicated, missing, or pointing nowhere, even when every word in the image is spelled correctly
- Text too small to read at 6.5 inches wide, or a label over about eight words
- An overall title or headline drawn above the panels: the figure carries none, only its panel titles
- A SPARSE CARTOON: a slogan panel (a title over a generic icon, a lone word or an empty box), no real number, no worked example, or a figure far sparser than the exemplars you kept
- A panel whose drawing does not show the claim its title makes, or panels out of the left-to-right order the specification gives
- A display equation or derivation, or a code snippet longer than a line or two
- Empty bands or wide margins the drawing does not fill, or a strip much wider than 16:9
- Overlapping, touching, cut-off or blurry labels, and poor contrast
- Font: serif labels in the style of LaTeX's Computer Modern, the paper's typeface, which the style preset asks for, to match the paper's other figures. Do not spend an attempt on the typeface alone when everything else is right

If ANY issue is found — even minor — do another attempt (within the 7-attempt limit). Do NOT accept a figure with problems as the final figure.

Change the prompt only when the prompt is what was wrong — a word you never specified, an element you forgot to name. For a defect the prompt already rules out, re-run it UNCHANGED: the same prompt sent twice gave a correct three-box chain once and four boxes with one label repeated the other time. Rewriting a prompt that was already right spends one of your 5 attempts on a variable that was not the cause.
</task>

<figure_specification>
Figure ID: fig1
Title: How five models represent coffee cooling
Caption: Five models for coffee cooling: (a) Newton's single-exponential law assumes constant cooling rate; (b) multi-regime models allow a transition between fast and slow cooling; (c) the Bayesian change-point detects the transition from residual structure. Blue marks multi-regime models, grey marks Newton's law.
Image Generation Description: Three panels left to right, each in a thin grey rounded frame on white, joined by thick dark arrows, filling the whole canvas. Panel (a) titled '(a) Newton\'s law': a cooling curve plot with x-axis 'Time (min) 0–30' and y-axis 'Temperature (°C) 20–95', showing a smooth blue exponential curve from 90°C to 40°C, and grey dots scattered around it with increasing deviation after 15 min; a label 'single-exponential: 2 parameters' and 'error peaks at 6.0°C at 30 min' in bold. Panel (b) titled '(b) Regime switching': the same axes, showing two curves — a blue piecewise curve with a visible kink at ~12 min labelled 'flux-crossover: \u03c4 = 5.5 min' and a green curve with a kink at ~10 min labelled 'piecewise: free transition' — both tracking the grey dots more closely; a label 'two regimes: fast then slow'. Panel (c) titled '(c) Change-point detection': the same axes, showing a red curve that adapts to the data with a kink at ~11 min labelled 'Bayesian change-point: detected from residuals' and 'RMSE 1.66°C vs 4.04°C'; a small inset showing the residual plot with a clear sign change at the transition point. Blue for multi-regime models, grey for Newton, red for Bayesian change-point in every panel. Style references: arXiv 2401.01234, 2305.06162.
Aspect Ratio: 16:9
Summary: Overview of the five cooling models and their representation of the regime transition
</figure_specification>

<critical_requirements>
1. Draw what the specification describes, adding only domain detail the specification, caption and summary support (a real token, a code line, a model name). Invent no result: no number they do not state, and no stage the method does not have
2. The concept rules above are hard rules: 2 to 4 lettered panels read left to right, no overall figure title, about 80 to 120 words with at least one real number, the field's own objects and no slogan panels, no display equations, a canvas no wider than 2:1
3. FONT: serif labels in the style of LaTeX's Computer Modern, the paper's typeface — `--style concept` asks for it on every call
4. Publication camera-ready style, in the manner of the exemplars you kept: white background, thin grey rounded panel frames, saturated colour that means something, no 3D effects/shadows/gradients. Follow aii-concept-fig-gen skill for image generation, prompting best practices, and figure type templates
5. TEXT SIZE: LARGE labels that never overlap or touch. Give a crowded figure room by shortening labels or dropping a secondary one, never by dropping the worked example or the real numbers, never by going wider and never by asking for smaller text
6. RESOLUTION: Explore layouts at 1K in the one phase above; no upscale or edit passes on this tier. Keep the aspect ratio from the spec. The paper's Figure 1 overview stays 16:9, never wider than 2:1
7. MANDATORY VERIFICATION: After EVERY generation attempt, read the image and check spelling first, then layout and meaning, then the print check (legibility and style) on the cropped deliverable. If anything is wrong, regenerate. Do NOT stop at the first clean-looking attempt
</critical_requirements>


<comparison_completeness>
If this figure's title, caption or summary names specific checkpoints, models or
variants being COMPARED — "ours vs baseline", "the base and the abliterated model",
"across the three checkpoints" — every one of them named there MUST appear in the
rendered figure as its own bar, curve, point or panel. Before you render, list every
comparator the specification names and check each one off as you draw it. A
comparison figure that quietly drops one of its own named comparators is wrong even
when every bar it does draw is numerically correct — the missing one is invisible to
anyone who was not told to look for it, which is what makes it worse than an
obviously incomplete figure.
</comparison_completeness>


<caption_from_the_rendered_figure>
The caption in <figure_specification> is a DRAFT, written before this figure existed by a step
that never saw it. After your final render, read the final image back and write the figure's
caption into the `caption` field of your output. It replaces the draft caption everywhere this
figure appears: the paper, the report and the paper's website.
- AT MOST 60 WORDS. One sentence on what the figure shows and what it argues, then only
  what a reader cannot see for themselves: what a colour or marker means where no legend says it,
  what the error bars are, the sample size. Do not re-list the axes, panels or labels the image
  already names; the detail belongs in the paper's text. A 300-word caption that walks the reader
  through every panel is the failure this limit exists to stop.
- Name a colour, marker, panel or series only if it is in the image. Where the draft caption and
  the image disagree (colours said to encode models when they encode languages, an axis the panel
  does not plot, a grey series that was never drawn), the image wins.
- Keep what is still true of the draft: the data, the sample size, what the error bars are, and
  the takeaway. Keep it LaTeX-ready in the same form as the draft caption.
- State no number the figure and its data do not carry.
- A figure with no data behind it that could pass for data (a drawn network, a map, a snapshot)
  is called a schematic illustration in its caption.
</caption_from_the_rendered_figure>


---

Output the result as JSON to: `/Volumes/aii-s20001/work/run_qZEiMJexNbcK/4_gen_paper_repo/_3_gen_viz/gen_viz_1/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "VizExpectedFiles": {
      "description": "Expected output files from viz generation.",
      "properties": {
        "image_path": {
          "description": "Path to the generated figure image file. Example: 'fig1_v0.jpg'",
          "title": "Image Path",
          "type": "string"
        }
      },
      "required": [
        "image_path"
      ],
      "title": "VizExpectedFiles",
      "type": "object"
    }
  },
  "description": "Structured output from viz figure generation agent.",
  "properties": {
    "title": {
      "description": "Figure title in plain, everyday language \u2014 short and jargon-free so a non-expert grasps it at a glance. Aim for about 4-8 words (~40 characters).",
      "maxLength": 90,
      "minLength": 12,
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "description": "Brief summary of the generated figure: what it shows, style, any issues fixed",
      "maxLength": 5000,
      "minLength": 500,
      "title": "Summary",
      "type": "string"
    },
    "caption": {
      "description": "The figure's caption, written from the FINAL rendered image after you read it back, in the same LaTeX-ready form as the draft caption. AT MOST ABOUT 60 WORDS: one sentence on what the figure shows and argues, then only what a reader cannot see for themselves (an encoding no legend states, what the error bars are, the sample size). Do not re-list the axes, panels or labels the image already names. It replaces the draft caption in the paper, the report and the paper's website. Name only axes, colours, markers, panels and series that are in the image.",
      "maxLength": 480,
      "minLength": 20,
      "title": "Caption",
      "type": "string"
    },
    "out_expected_files": {
      "$ref": "#/$defs/VizExpectedFiles",
      "description": "Output file you created. Must include the generated figure image path."
    }
  },
  "required": [
    "title",
    "summary",
    "caption",
    "out_expected_files"
  ],
  "title": "VizFigureOutput",
  "type": "object"
}
```

IMPORTANT: this task is NOT complete until `/Volumes/aii-s20001/work/run_qZEiMJexNbcK/4_gen_paper_repo/_3_gen_viz/gen_viz_1/.sdk_openhands_agent_struct_out.json` exists and contains JSON matching the schema above.
````

### [5] HUMAN-USER prompt · 2026-11-24 21:07:01 UTC

```
How accurately does Newton's law of cooling predict the temperature of a cup of coffee over 30 minutes, compared with published measurements?
```

### [6] HUMAN-USER prompt · 2026-11-24 21:07:01 UTC

```
Continue from where you stopped.
```

### [7] SYSTEM-USER prompt · 2026-11-24 21:24:31 UTC

```
<validation-feedback>
Attempt 1 failed validation.

Schema validation found 1 problem — fix ALL of them at once:
  - at `caption`: "Five models for coffee cooling. (a) Newton's single-exponential law assumes constant cooling rate; the blue curve deviates increasingly from measurements after ~15 min, peaking at an error of 6.0°C at 30 min. (b) Multi-regime models allow a transition between fast and slow cooling: flux-crossover (blue, τ = 5.5 min) and piecewise (green, free transition) both track the data more closely. (c) The Bayesian change-point (red) is detected from residuals, yielding RMSE 1.66°C vs 4.04°C for Newton. Grey points are observations; blue/green denote multi-regime models; red denotes the Bayesian fit." is too long (at most 480 characters, got 596)
Every required field must be present and every field type must match the schema.

Produce `/Volumes/aii-s20001/work/run_qZEiMJexNbcK/4_gen_paper_repo/_3_gen_viz/gen_viz_1/.sdk_openhands_agent_struct_out.json` again so it contains corrected JSON that matches the schema. Do not invent new fields.
</validation-feedback>
```

### [8] SYSTEM-USER prompt · 2026-11-24 21:25:40 UTC

```
<CRITICAL_ERROR>
The module-end file check FAILED (attempt 1/3).

PROBLEMS:
  - .aii/manifest.yaml: 'fig1_all/' matches nothing that needs a decision — remove it (text, code and files under the auto-keep floor are always kept)

FIX IT:
1. Add one entry per uncovered path to `.aii/manifest.yaml` (create it if missing).
   Every path is RELATIVE TO YOUR CWD and must resolve inside it. Globs and
   whole directories are fine — a whole `hf_cache/` is ONE entry.

   entries:
     - path: results/
       keep: six GPU-hours of sweep output, not reproducible in this run
     - path: hf_cache/
       delete: redownloadable
       source: "huggingface-cli download meta-llama/Llama-3-8B"
     - path: checkpoints/
       delete: regenerable
       source: "uv run train.py --epochs 3"

   `keep:` takes a one-line reason. `delete:` takes `redownloadable` or
   `regenerable` and a `source:` that brings the files back.
2. Make sure `README.md` reads like a GitHub repository README: what you did,
   the layout (a line per important file/dir), how to run it, and a
   "Restoring removed files" section with the command for EVERY delete entry.
3. Text and code files never need a decision, and neither does anything under
   the auto-keep floor. Only large binaries and cache directories do.
</CRITICAL_ERROR>
```
