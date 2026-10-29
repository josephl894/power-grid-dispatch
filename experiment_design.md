# OUTDATED

This document reflects the older PGLib/UnitCommitment.jl project framing. The
final submission has moved to a public-data NYISO SCUC proxy using Gold
Book/P-19 capacity, P-32 interfaces, P-511A/P-54A/P-6B stress features, and
A5b/A6c/A7b ablations.

# Experiment Design Proposal
### Teaching a Fast Power-Dispatch Solver to Imitate a Slow, Uncertainty-Aware One

---

## The real-world problem we're solving

Every day, grid operators have to answer two questions for the next 24 hours: **which power plants should be running, and how much electricity should each one produce?** This is called *unit commitment* (UC).

They have to make that plan before they know exactly what will happen — tomorrow's electricity demand is uncertain, and if the grid has wind or solar, tomorrow's renewable output is even more uncertain. Making a bad plan is costly: if demand is higher than expected and your slow-starting plants are off, you pay huge shortfall penalties. If renewables surprise you on the upside, you've wasted money starting expensive plants you didn't need.

There are two standard ways to make this plan:

- **Stochastic UC (accurate but slow).** Run a solver that considers many possible futures ("scenarios") all at once and picks the plan with lowest expected cost. This takes the uncertainty seriously, but it's slow — scales poorly as the grid gets bigger or the scenario count grows.
- **Deterministic UC (fast but dumb).** Run a solver that only considers one "average" future. Fast, but it treats uncertainty as if it didn't exist, so its plans can be fragile.

The research idea is: **can we train a small model that teaches the fast solver to make plans almost as good as the slow solver's?**

The trick is to *adjust the cost numbers* the fast solver uses. Instead of feeding the fast solver the literal fuel cost of each plant, we feed it a *learned* cost that encodes "this plant is more expensive to schedule today because forecasts are uncertain" or "this plant is cheaper today because the wind is steady." The plant's real physical cost doesn't change, but the solver's behavior does.

---

## Research question (plain English)

> **Can we make a fast, simple power-dispatch solver produce plans almost as good as a slow, uncertainty-aware one by teaching it smarter cost numbers? And how much does it matter which training loss we use, and whether we include weather data?**

The whole project comes down to that question. Two knobs to test: the training loss (a simple regression loss versus a more principled structured loss) and the feature set (only what's already in the problem data versus also including weather).

---

## What's in and what's out (reduction first)

**In scope**
- One real application: day-ahead unit commitment with renewables.
- One source of problem data: PGLib-UC benchmark instances.
- One slow solver (as the teacher): UnitCommitment.jl in stochastic mode.
- One fast solver (as the student): UnitCommitment.jl in deterministic mode, with our learned costs.
- Three methods to compare: naive baseline, regression-trained, structured-loss-trained.
- One main number we report: how costly is each method's plan when the uncertainty actually plays out?

**Deliberately cut**
- CVaR and risk measures — we just minimize expected cost.
- Writing our own solver — we don't.
- The Boltzmann / partition-function math from the original proposal — replaced by Fenchel-Young loss.
- Synthetic data generation pipelines — we use existing benchmark problems.
- Scaling to big grids — we stay on small instances (≤30 plants).
- Neural networks — our model is a simple linear combination of a few features.

---

## Data sources (researched and verified)

### Problem instances — PGLib-UC
What it is: a library of realistic unit-commitment test problems in JSON format. Each file describes one day: a list of power plants with their physical parameters (minimum/maximum output, ramp rates, startup costs, fuel cost curves), a 48-hour demand forecast, and a 48-hour renewable forecast.

- Source: https://github.com/power-grid-lib/pglib-uc
- License: Creative Commons Attribution, course-project compatible.
- Subset we use: `rts_gmlc/` — around 30 thermal plants per instance, plus wind and solar units.
- Enough distinct daily instances to support a train/validation/test split of roughly 150/25/25.

### Teacher solver — UnitCommitment.jl
What it does: takes a UC problem in JSON form and solves it. Supports both deterministic mode (one "average" future) and stochastic mode (many scenarios jointly). The stochastic mode is our teacher; the deterministic mode is our student.

- Source: https://github.com/ANL-CEEESA/UnitCommitment.jl
- Stochastic mode verified in the repo's README.
- Julia/JuMP package, installed via Julia's package manager.

### Making problems stochastic
PGLib-UC instances ship deterministic. We create uncertainty by sampling 20 perturbed versions of each instance's demand and renewable forecasts:
- Demand: small Gaussian noise (around 3–5% std) with hour-to-hour correlation.
- Renewable output: larger noise (10–20% std) with hour-to-hour correlation.

Those 20 perturbations are the "scenarios" the stochastic teacher sees.

### Weather data (Phase 2, optional)
Each renewable plant in an RTS-GMLC-based instance has a name like `101_PV_1`, where `101` is a bus ID. RTS-GMLC's `bus.csv` file gives the lat/lng of each bus. Once we have coordinates, we can query:

- **NREL WIND Toolkit** (https://developer.nrel.gov/docs/wind/wind-toolkit/) for wind speed at turbine height.
- **NREL NSRDB** (https://developer.nrel.gov/docs/solar/nsrdb/) for solar irradiance.

Both are free, have documented APIs, and accept point queries by (lat, lng, timestamp).

---

## The training pipeline

For each training problem instance:

1. **Sample 20 scenarios** of perturbed demand and renewables.
2. **Run the slow (stochastic) solver** on all 20 scenarios → get the "correct" plan, which we call the oracle.
3. **Compute features** φ for each (plant, hour) pair — a small set of numbers summarizing the plant's parameters and the day's conditions.
4. **Compute learned costs** ĉ = θᵀφ, where θ is a small weight vector we're training.
5. **Run the fast (deterministic) solver** with costs ĉ → get a predicted plan.
6. **Compare predicted plan to oracle**, take a gradient step on θ to bring them closer.
7. Repeat over all training instances.

After training, we have a small weight vector θ that encodes "how to turn day-specific context into cost adjustments that make the fast solver mimic the slow one."

At test time, the pipeline is just: new problem → compute features → multiply by θ → run fast solver once. Seconds, not minutes.

---

## The three methods we compare

1. **Naive baseline.** Run the fast solver with the instance's raw cost coefficients. No learning. This is what a grid operator does today without ML.
2. **Cost regression.** Train θ by ordinary least squares — regress the learned costs onto the "marginal costs" implied by the oracle plan. Simple, well-understood.
3. **Fenchel-Young loss.** Train θ by directly comparing the fast solver's plan to the oracle's plan using a Fenchel-Young loss. More principled for this kind of structured prediction problem; the professor specifically suggested it.

Each method is run twice — once with instance-only features, once with weather features added — so we can see whether weather data actually helps.

---

## Features

### Instance-only (the minimum set, Phase 1)
For each plant i at hour t, we build around 15 numbers:
- Plant parameters (max capacity, min capacity, ramp rate, fuel cost coefficient, startup cost).
- Time indicators (hour of day, weekday).
- Demand at hour t (nominal value, plus mean and std across the 20 scenarios).
- Renewable output at hour t (nominal, plus mean and std across scenarios).
- Net load (demand minus renewable) at hour t.

### Weather-augmented (Phase 2, adds ~7 more dimensions)
For each renewable plant, we add:
- Wind speed at 100m — mean, std, hour-to-hour ramp (for wind plants).
- Solar irradiance — mean, std, hour-to-hour ramp (for solar plants).

---

## How we grade success

For each test instance:

- Build three plans (naive, regression-trained, FY-trained) using the fast solver.
- Build the oracle plan using the slow solver.
- Roll a fresh set of 50 "real" scenarios to see what actually happens.
- For each plan, compute the realized cost: fuel costs + startup costs + any shortfall penalties when the plan couldn't cover actual demand.

We report:
- Mean realized cost, relative to the oracle (how close did we get?) and relative to the naive baseline (how much did learning help?).
- Runtime per decision (how much faster is our method than the oracle?).
- With- and without-weather results (does weather data actually help?).

---

## Engineering plan

**Phase 1 — core pipeline (Weeks 1–2).**
Get one end-to-end run working on a single small instance. Load the JSON, sample scenarios, run the slow solver, extract features, train θ by regression, run the fast solver, compare plans. Correctness matters, speed doesn't.

**Phase 2 — scale up, add Fenchel-Young (Week 3).**
Run on the full instance set. Implement the FY loss as a second training method. Produce the main comparison table.

**Phase 3 — weather features (Week 4).**
Join each renewable plant to a lat/lng via `bus.csv`. Query NREL APIs and cache responses locally. Rerun the comparison with weather-augmented features. Produce the ablation table.

**Phase 4 — writeup (Week 5).**
Draft the final report around the two result tables. Discuss what worked, what didn't, what's still open.

---

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| The stochastic solver is hard to configure | Drop back to openSDUC (Python, readable). Worst case, generate oracles by solving deterministic UC with scenario-weighted expected costs — coarser but still useful. |
| Fenchel-Young implementation is brittle | Make regression the primary result; report FY as a secondary discussion. |
| NREL API calls fail or hit rate limits | Cache aggressively; if blocked entirely, drop weather features and report the instance-only result as final. |
| Training on all instances takes too long | Drop to 50 training instances. The headline comparison still works. |
| Linear θ underfits | Report the underfit openly — it becomes a finding about what simple LtO surrogates can and can't do. |

---

## Why this fits the course and the feedback

This is a benchmarking study in the context of an application (Archetype 2 from the project description). It is grounded in end-to-end learning and optimization: we're learning a cost function specifically so that a downstream solver produces good decisions. It directly responds to every piece of the professor's feedback — one focused research question, no Boltzmann machinery, no home-brewed solver, public benchmarks, Fenchel-Young loss as suggested, and a clearer baseline than the previous version.
