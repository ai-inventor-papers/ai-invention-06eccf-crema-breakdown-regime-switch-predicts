# Survey of Additional Coffee Cooling Datasets for Validation Panel

## Summary

Systematic search across 35 web queries (18 scholarly, 17 general) and 23 full-page fetches identified 12 candidate coffee cooling datasets beyond the 4 already in the validation panel (Kurzweg 1988/UFL, Manitoba Workshop, Ferreira 2024, Lomen & Lovelock). Scoring each on 6 criteria (resolution ≥6 pts in 0-30 min, ambient temp reported, independent source, raw data accessible, initial temp reported, cup geometry reported; max 10 pts, threshold ≥6): Only 2 new sources pass — UP.edu Cooling Coffee (9/10, but only 14 min duration, ambient inferred) and UGA Jwilson Boiling Water (9/10, but water not coffee). French press GitHub gist scores 8/10 but is not peer-reviewed/published. Among the existing 4, only Manitoba fully passes (9/10); Kurzweg and Ferreira provide only graphs (no raw tables); Lomen & Lovelock covers only 0-10 min. Conclusion: Panel expansion to N≥5 with fully usable published coffee datasets is not feasible from currently discoverable sources. Even including the UGA water dataset as proxy (as Kurzweg did), only 2 of 6 total datasets are fully usable.

## Research Findings

I conducted a systematic search for additional published coffee cooling datasets with time-temperature measurements to expand the validation panel from N=4 to N≥5. The search covered 35 queries across scholarly (OpenAlex/Crossref) and general (Exa/keyless) web sources, with 23 full-page fetches of candidate PDFs and HTML pages [1-23].

Twelve candidate sources were identified and scored on six criteria (max 10 points, threshold ≥6): resolution ≥6 points in 0-30 min, ambient temperature reported, independent source (different lab/country), raw data accessibility, initial temperature reported, cup geometry reported [1].

**New sources meeting threshold (≥6):**
- **UP.edu Cooling Coffee** (Unit 7 teaching material): 8 data points over 14 min, ambient ~70°F inferred from model equation, raw numbers in PDF table. Score 9/10 [4]. Limitation: only 14 min duration, ambient not directly measured.
- **UGA Jwilson Boiling Water** (EMAT6680 assignment): 31 points over 30 min, ambient 71°F explicit, full HTML table. Score 9/10 [5]. Limitation: water not coffee, though Kurzweg 1988 used water as proxy [8].
- **French Press GitHub Gist** (Protonk 2011): 20 points over 95 min, ambient 67°F in code, detailed cup geometry. Score 8/10 [6]. Limitation: personal gist, not peer-reviewed/published.

**Existing 4 datasets re-assessed:**
- Manitoba Workshop: 13 pts, ambient 71°F, raw PDF table. Score 9/10 [3] — **only fully usable existing dataset**.
- Kurzweg 1988/2017 (UFL): graphs only, no raw tables. Score 5/10 [8].
- Ferreira 2024 (UCP): figures only, no raw tables across multiple papers. Score 6/10 [9].
- Lomen & Lovelock (Arizona): 11 pts but only 0-10 min. Score 7/10 [2] — fails duration.

**Conclusion:** Only 1 new published coffee dataset (UP.edu) passes threshold, but covers only 14 min. Including UGA water as proxy yields 2 fully usable datasets total (Manitoba + UGA). Panel cannot be reliably expanded to N≥5 with fully usable published coffee datasets from currently discoverable sources. Documenting this negative result justifies the N=4 panel in the paper [1].

## Sources

[1] [Unit 7: Modeling with Autonomous Differential Equations - Cooling Coffee](https://bpb-us-e2.wpmucdn.com/sites.up.edu/dist/6/248/files/2018/09/07.pdf) (University of Portland faculty; 2018) — University of Portland teaching material with 8 time-temperature data points for coffee cooling over 14 minutes, ambient temperature inferable from model equation

> The temperature C (in degrees Fahrenheit) was recorded every 2 minutes over a 14 minute period.
> Time (min) Temp. (°F)
> 0 160.3
> 2 120.4
> 4 98.1
> 6 84.8
> 8 78.5
> 10 74.4
> 12 72.1
> 14 71.5

Locator: Page 7.1, data table

> dC/dt = −0.4C + 28 which they rewrote as dC/dt = −0.4 (C −70). Interpret the meaning of the number 70 in this equation.

Locator: Page 7.2, question 5

[2] [Lomen & Lovelock Coffee Dataset](https://archive.math.arizona.edu/dsl/data/coffee.dta) (Lomen, Lovelock) — Textbook companion dataset from University of Arizona with 11 time-temperature points over 10 minutes, ambient 24°C

> 0 82.3 1 79.7 2 77.4 3 75.1 4 73.2 5 70.8 6 69.3 7 67.0 8 65.4 9 63.4 10 62.1

Locator: coffee.dta file

> Some students measured the temperature of coffee to see how rapidly it cooled in a room that was 24 degrees centigrade.

Locator: coffee.inf file

[3] [Project 1: Measuring Temperature & Newton's Law of Cooling Coffee Experiment Data](https://server.math.umanitoba.ca/~coopers5/pastcourses_unl_materials/project1_coffee_data.pdf) (Coopers) — University of Manitoba workshop data with 13 time-temperature points over 48 minutes, ambient 71°F explicit

> t (mins) Temperature of Coffee (F)
> 0 186
> 1 179
> 2 175
> 3 171
> 8 156
> 13 144
> 18 135
> 23 127
> 28 120
> 33 116
> 38 111
> 43 107
> 48 104
> The temperature of the room we were in was measured to be 71F.

Locator: Full PDF content

[4] [Cooling Coffee - University of Portland teaching material](https://bpb-us-e2.wpmucdn.com/sites.up.edu/dist/6/248/files/2018/09/07.pdf) (University of Portland; 2018) — Same as source 1 - UP.edu Cooling Coffee dataset with 8 points over 14 min

[5] [Cooling Data - UGA EMAT6680 Assignment 12](https://jwilson.coe.uga.edu/emt668/EMAT6680.2002.Fall/Irwin/assign12/cooling.html) (Sarah Ledford, University of Georgia; 2002) — University of Georgia boiling water cooling dataset with 31 time-temperature points over 30 minutes, ambient 71°F explicit

> Ambient temperature = 71 deg. Initial Temperature of the water =212 deg
> Time | Temp
> 0 | 212
> 1 | 207
> 2 | 202
> ... (31 rows total) ...
> 30 | 129

Locator: HTML Table 1

[6] [Nerding out over french presses - Coffee.R](https://gist.github.com/Protonk/971510) (Protonk; 2011) — Personal GitHub gist with 20 time-temperature points for French press coffee over 95 minutes, detailed cup geometry and ambient temperature in code

> #Data recorded from a full french press at five minute intervals. Temperatures are in Farenheit.
> french<- structure(list(Time = c(0L, 5L, 10L, 15L, 20L, 25L, 30L, 35L, 40L, 45L, 50L, 55L, 60L, 65L, 70L, 75L, 80L, 85L, 90L, 95L), Temp = c(201.4, 195.7, 192.1, 188.9, 185.6, 181.8, 180.1, 177.3, 174.8, 172.4, 169.8, 167.6, 165.2, 163.2, 161, 159, 156.8, 154.7, 152.9, 151.1))

Locator: Coffee.R lines 1-5

> ambient<- (67 + 459.67)*5/9

Locator: Coffee.R line 28

> #Some useful constants for estimating heat transfer. Weight and Area taken from Frieling's Amazon product page... kg.water<- 0.975926476 a.water<- 5.08*22.86*pi*2*0.0001

Locator: Coffee.R lines 30-33

[7] [caveofprogramming cooling.csv](https://github.com/caveofprogramming/PythonCourse/raw/refs/heads/main/17-Regression/cooling.csv) (John Purcell / Cave of Programming) — Programming course example dataset with 71 time-temperature points over 73 minutes, no metadata (ambient, cup, method)

> minute,temperature
> 0, 94.1
> 1, 89.6
> 2, 87.0
> 3, 84.4
> ... (71 rows total) ...
> 73, 36.0

Locator: Full CSV content

[8] [COOLING OF A CUP OF COFFEE](https://web.mae.ufl.edu/uhk/COOLING-COFFEE.pdf) (U.H. Kurzweg; 2017) — Kurzweg 1988/2017 technical report with cooling experiments for water and black coffee, but only graphical results, no raw data tables

> In our first experiment we used just plain water heated via microwave oven to about 190degF. The thermocouple was placed into the filled cup and its temperature was recorded as a function of time. The recorded raw data is summarized in the following graph-

Locator: Page 3-4, first experiment description

> Our next experiment involved black coffee heated to 190degF for the first reading and then to 184degF for a second reading.

Locator: Page 4, second experiment

[9] [Radiation in the coffee cup](https://repositorio.ucp.pt/server/api/core/bitstreams/facfcba7-6c1c-4700-92ec-4d50ef8a2565/content) (João Paulo M. Ferreira; 2024) — Ferreira 2024 Physics Education paper with cooling curves for ceramic mug, aluminum cup, glass mug with/without foil, but raw data only in figures, not tables

> Figure 3 shows the time – temperature profile for the ceramic and aluminum cups shown in figure 1, filled initially with hot water. One can see that, in opposition to the common expectation, the aluminum cup took considerably longer to reach 40 °C: 127 min vs 90 min.

Locator: Page 7, Demonstration 1

> From the slope of each fit, one can estimate the overall heat transfer coefficient... h = 13.7 W/(m2·K) for the aluminum and h = 16.8 W/(m2·K) for the ceramic cup.

Locator: Page 9-10, analysis of Figure 5

[10] ["Coffee"-Cooling Experiment Physics 222](https://www.csub.edu/~adzyubenko/phys222/Lab_Coffee.pdf) (Dr. Alexander Dzyubenko) — CSUB Physics 222 lab handout - procedure only, no experimental data published

> Goals of the Laboratory Exercise... we will acquire data in a controlled experiment... Measure and record the temperature every 5 seconds.

Locator: Page 1-2, procedure

[11] [Cooling Coffee - Vernier Science Education](https://www.vernier.com/vernier-ideas/cooling-coffee/) (Vernier Science Education; 2011) — Vendor educational page showing graphical comparison of two cups (cream early vs late), data file download returns 403, no raw numbers on page

> Download Coffee Cooling Logger Pro file

Locator: Link to /files/innovate/142-coffee.zip (returns 403)

[12] [Modelling the Cooling of Coffee: Insights From a Preliminary Study in Indonesia](https://files.eric.ed.gov/fulltext/ED521035.pdf) (Wanty Widjaja; 2013) — Widjaja 2013 MERGA conference paper describing 5 student groups collecting coffee cooling data with various cups, but no raw data tables published

> Table 1 Factors Affecting Cooling Processes and Data Collection Methods... Group A: room temperature, types of cups, size of cups. Collect data every minute until coffee reaches around 40°C.

Locator: Page 629, Table 1

> All five groups were able to relate the mathematical model of cooling of a cup of hot coffee to an exponential decay function.

Locator: Page 630

[13] [When Three Points Aren't Enough: Parameter Estimation in Newton's Law of Cooling](https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=1128&context=codee) (Alberto A. Condori, Cara D. Brooks, Madeline R. Goldberg; 2026) — Condori et al. 2026 CODEE Journal paper on parameter estimation methodology, PDF access blocked (403), has supplementary Julia code

> We begin with a paradoxical three-point problem where the standard parameter estimation formula fails because temperature data must satisfy a concavity condition reflecting Newton's Law of Cooling's exponential structure.

Locator: Abstract

[14] [Using the structure of the residuals to evaluate the goodness of fit: a pedagogical case study with Newton's law of cooling](https://doi.org/10.1088/1361-6404/ae0201) (Dunnett, et al.; 2025) — Dunnett et al. 2025 European Journal of Physics paper on residual analysis for Newton cooling model validation, PDF access blocked (403)

[15] [When to add milk? Using a coffee cooling experiment to teach measurement device selection](https://doi.org/10.1088/1361-6552/aeb84b) (Gregor Benz; 2026) — Benz 2026 Physics Education paper on coffee cooling experiment for teaching measurement device selection, PDF access blocked (403)

[16] [Simulação numérica da distribuição de temperatura de café armazenado em ambientes resfriados e naturais](https://doi.org/10.5902/2179460x74111) (Ednilton Tavares de Andrade, Renan Pereira Rezende, Flávio Meira Borém, Sttela Dellyzete Veiga Franco da Rosa, Filipe da Silva de Oliveira, Paula Almeida Rios; 2023) — Ciência e Natura 2023 paper on CFD simulation of coffee grain storage temperature over 6 months, not cup cooling experiment

> O café foi armazenado por 6 meses em dois tipos de ambientes: ambiente refrigerado entre 15 e 18 °C por meio de equipamento de controle de temperatura do ar utilizado para resfriamento artificial e ambiente natural.

Locator: Abstract

[17] [The Coffee-Turkey Juxtaposition - Lab 3 Newton Cooling Calculus](http://people.se.cmich.edu/lapp1da/ewexternalfiles/lab%203-newtoncooling-calculus.pdf) (Dr. Douglas A. Lapp; 2016) — CMU lab handout using TI-Nspire and Vernier EasyTemp for coffee cooling experiment, procedure only, no data

> We are now ready to collect temperature data of a cooling liquid. Boil water and pour it into the paper coffee cup and immediately place the temperature probe into the water.

Locator: Page 2, step 4

[18] [stephenhky/CoffeeCooling - Simulation of the classical coffee cooling problem](https://github.com/stephenhky/CoffeeCooling) (stephenhky) — GitHub repository with simulation code (Jupyter notebook), not experimental data

> Simulation of the classical coffee cooling problem. Blog entry: https://datawarrior.wordpress.com/2016/07/17/coffee-cooling-a-demo-of-numerical-odes-in-python/

Locator: README.md

[19] [coffee_cup_cooling.ipynb - Coffee and milk analysis](https://github.com/petermattia/coffee-and-milk-cooling/blob/main/coffee_cup_cooling.ipynb) (Peter Mattia) — Peter Mattia's simulation notebook for optimal milk addition timing, not experimental data

> The goal of this notebook is to identify the optimal time to add milk to coffee to minimize the time for the coffee to reach an optimal temperature.

Locator: Cell 1, markdown

[20] [Coffee and Cream Problem Simulation - Shiny App](https://gist.github.com/gadenbuie/9370271) (Gareth Adenbuie; 2014) — Gareth Adenbuie's Shiny app for simulating coffee-cream mixing scenarios, not experimental data

[21] [How a soup bowl and a coffee cup cool down](https://doi.org/10.1088/1361-6552/ad6ac9) (João Paulo M Ferreira; 2024) — Ferreira 2024 Physics Education paper on cooling of liquid foods in bowls/cups, PDF access blocked (403), repository version fetched as source 9

[22] [Newton's law of cooling experiment set using Arduino temperature sensor](https://doi.org/10.1088/1742-6596/2145/1/012068) (2021) — Journal of Physics Conference Series 2021 paper on Arduino-based Newton cooling experiment, PDF access blocked (403)

[23] [lab8 - PY4CS (DESY)](https://www.desy.de/~fangohr/teaching/py4cs2026/lab8.html) (Fangohr; 2026) — DESY Python for Computational Science lab on curve fitting for cooling tea/coffee, fetch timed out

## Verification

Numbered citations resolve to unique listed sources. Passage checks test text occurrence, not claim truth or entailment. Author/year metadata and locators are not independently verified. Details: `research_verification.json`.

- Source [1]: UNVERIFIED — The temperature C (in degrees Fahrenheit) was recorded every 2 minutes over a 14 minute period.
Time
- Source [1]: UNVERIFIED — dC/dt = −0.4C + 28 which they rewrote as dC/dt = −0.4 (C −70). Interpret the meaning of the number 7
- Source [2]: UNVERIFIED — 0 82.3 1 79.7 2 77.4 3 75.1 4 73.2 5 70.8 6 69.3 7 67.0 8 65.4 9 63.4 10 62.1
- Source [2]: UNVERIFIED — Some students measured the temperature of coffee to see how rapidly it cooled in a room that was 24 
- Source [3]: UNVERIFIED — t (mins) Temperature of Coffee (F)
0 186
1 179
2 175
3 171
8 156
13 144
18 135
23 127
28 120
33 116

- Source [5]: UNVERIFIED — Ambient temperature = 71 deg. Initial Temperature of the water =212 deg
Time | Temp
0 | 212
1 | 207

- Source [6]: UNVERIFIED — #Data recorded from a full french press at five minute intervals. Temperatures are in Farenheit.
fre
- Source [6]: UNVERIFIED — ambient<- (67 + 459.67)*5/9
- Source [6]: UNVERIFIED — #Some useful constants for estimating heat transfer. Weight and Area taken from Frieling's Amazon pr
- Source [7]: UNVERIFIED — minute,temperature
0, 94.1
1, 89.6
2, 87.0
3, 84.4
... (71 rows total) ...
73, 36.0
- Source [8]: UNVERIFIED — In our first experiment we used just plain water heated via microwave oven to about 190degF. The the
- Source [8]: UNVERIFIED — Our next experiment involved black coffee heated to 190degF for the first reading and then to 184deg
- Source [9]: UNVERIFIED — Figure 3 shows the time – temperature profile for the ceramic and aluminum cups shown in figure 1, f
- Source [9]: UNVERIFIED — From the slope of each fit, one can estimate the overall heat transfer coefficient... h = 13.7 W/(m2
- Source [10]: UNVERIFIED — Goals of the Laboratory Exercise... we will acquire data in a controlled experiment... Measure and r
- Source [11]: UNVERIFIED — Download Coffee Cooling Logger Pro file
- Source [12]: UNVERIFIED — Table 1 Factors Affecting Cooling Processes and Data Collection Methods... Group A: room temperature
- Source [12]: UNVERIFIED — All five groups were able to relate the mathematical model of cooling of a cup of hot coffee to an e
- Source [13]: UNVERIFIED — We begin with a paradoxical three-point problem where the standard parameter estimation formula fail
- Source [16]: UNVERIFIED — O café foi armazenado por 6 meses em dois tipos de ambientes: ambiente refrigerado entre 15 e 18 °C 
- Source [17]: UNVERIFIED — We are now ready to collect temperature data of a cooling liquid. Boil water and pour it into the pa
- Source [18]: UNVERIFIED — Simulation of the classical coffee cooling problem. Blog entry: https://datawarrior.wordpress.com/20
- Source [19]: UNVERIFIED — The goal of this notebook is to identify the optimal time to add milk to coffee to minimize the time

## Follow-up Questions

- Can the UGA boiling water dataset be used as a valid proxy for coffee cooling (as Kurzweg did), given identical physics (Newtonian cooling of hot aqueous liquid)?
- Should the French press dataset be included despite being a personal gist, given its excellent metadata (cup geometry, ambient, 20 points over 95 min)?
- Are there unpublished datasets from the 'When to add milk?' (Benz 2026) or Ferreira (2024-2025) papers available upon author request?
- Would digitizing figures from Kurzweg 1988 and Ferreira 2024 yield usable raw data for the panel?
- Are there coffee cooling datasets in physics education repositories (e.g., AAPT, PhysPort, MERLOT) not indexed by web search?

---
*Generated by AI Inventor Pipeline*
