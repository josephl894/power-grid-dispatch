# Research Influence and Design Decisions

This note records the main design decisions for the NYISO cost-aware dispatch pivot and the sources that motivated them. The emphasis is on the two methods we are carrying forward:

- in-network Fenchel-Young / learning-to-optimize dispatch, where the learned objective is inside the optimization layer;
- out-of-network linear repair dispatch, where a predictor proposes dispatch and an LP repairs it.

## Research Framing

- We are treating the project as a public-data, financially informed dispatch proxy study, not a full NYISO unit-commitment reconstruction.
  - The current code models NYISO fuel dispatch using public NYISO load, price, constraint, outage, weather, and generator metadata data, with P-63 fuel mix as the observed dispatch signal. Source: `design_descision.md`, `nyiso_processing.py`, `nyiso_experiments.py`.
  - The OPF and SCED literature frames dispatch as choosing generator setpoints at minimum production cost subject to physical and operational constraints. Sources:
    - `Literature Review/Energy Dispatch Optimisation Literature/Machine Learning for Optimal Power Flows.pdf`
    - `Literature Review/Energy Dispatch Optimisation Literature/Learning optimization proxies or large-scale Security Constrained Economic Dispatch.pdf`
    - `Literature Review/Energy Dispatch Optimisation Literature/End-to-End Feasible Optimization Proxies for Large-Scale Economic ispatch.pdf`
  - We cannot claim true NYISO production-cost optimality because public data does not expose complete private generator offer curves, full commitment schedules, network-security models, or settlement-side information.

- We are moving from system-fuel dispatch toward zone-fuel dispatch.
  - Current system model has 7 fuel categories and a copperplate/system-level LP. Source: `design_descision.md`.
  - Zone-fuel dispatch is the next tractable step: 11 NYISO zones times 7 fuel categories, preserving the LtO structure while adding spatial detail.
  - This is more defensible than immediate generator-level UC because public generator bid data is masked and cannot be cleanly joined to PTID/fuel/zone metadata.

- We retain NYISO observed dispatch as a realism benchmark, not as the sole training objective.
  - The previous model was primarily imitation-oriented: learn costs or residual adjustments that mimic P-63 fuel dispatch. Source: `nyiso_experiments.py`.
  - The pivot makes financial cost/regret the main objective, while dispatch distance to P-63 remains a diagnostic for missing constraints or implausible solutions.

## Cost Objective

- The economic dispatch objective should minimize production cost plus feasibility penalties.
  - Literature abstracts generator costs as `c_g(p_g)` or production-cost functions in ED/SCED/OPF formulations. Sources:
    - `End-to-End Feasible Optimization Proxies for Large-Scale Economic Dispatch.pdf`: uses generator production-cost functions and self-supervised objective minimization.
    - `Learning optimization proxies for large-scale Security-Constrained Economic Dispatch.pdf`: describes production costs as piecewise-linear convex functions submitted through market offers.
    - `Machine Learning for Optimal Power Flows.pdf`: formulates OPF as minimizing generator cost subject to network constraints.

- For public-data cost construction, we use the standard thermal variable-cost proxy:
  - `variable_cost_$per_MWh = fuel_price_$per_MMBtu * heat_rate_MMBtu_per_MWh + variable_OM_$per_MWh`.
  - This instantiates the abstract production-cost function when true generator offer curves are unavailable.
  - Unit logic:
    - fuel price: dollars per MMBtu;
    - heat rate: MMBtu per MWh;
    - variable O&M: dollars per MWh.
  - Online sources:
    - EIA heat-rate definition: https://www.eia.gov/tools/faqs/faq.php?id=107&t=1
    - EIA-923 data: https://www.eia.gov/electricity/data/eia923/
    - NREL ATB definitions/data: https://atb.nrel.gov/electricity/

- We separate engineering production cost from market price.
  - Fuel price and heat rate approximate generator variable production cost.
  - LBMP is a market-clearing price and should not be treated as the generator's private cost.
  - P-2A/P-24A zonal LBMP remain useful as price, congestion, and realized-market evaluation signals.
  - P-2B/P-24B/P-4B generator LBMP would improve generator-level price resolution if populated, but they still do not reveal private cost curves.

- P-27 bid data is useful but not the clean cost truth.
  - P-27 Gen contains masked generator bids, dispatch MW/price segments, fixed min-gen costs, reserve offers, and regulation offers.
  - P-27 UC contains masked min-run, min-down, max stops, startup times, and startup costs.
  - Because resource IDs are masked, P-27 cannot be assumed to join to actual NYISO generator PTIDs, zones, fuel types, or capacities.
  - Design use:
    - market-wide offer-distribution features;
    - startup/ramping friction proxies;
    - sanity checks for cost levels;
    - not the primary zone-fuel cost oracle.

## Method 1: In-Network FY LtO Model

- The in-network LtO model should learn dispatch-driving costs, not just imitation costs.
  - Existing code maps features to cost vectors and solves an LP: `c_t = h_theta(X_t)`, `g_hat_t = argmin_g c_t^T g`. Source: `nyiso_experiments.py`, `StructuredCostModel`, `ResidualAwareCostModel`.
  - The previous target is projected observed dispatch. The pivot should train toward cost-optimal feasible dispatch or financial regret under a public-data cost stack.

- The FY / structured loss remains appropriate as a structured prediction surrogate.
  - The FY-style loss compares the oracle feasible solution and the model-induced solution through the learned objective.
  - Source influence:
    - original proposal and LtO framing: `Project Proposal.pdf`
    - task-based end-to-end optimization: `Literature Review/Energy Dispatch Optimisation Literature/Task-based End-to-end Model Learning in Stochastic Optimization.pdf`
    - feasible optimization proxy / self-supervised ED learning: `End-to-End Feasible Optimization Proxies for Large-Scale Economic Dispatch.pdf`

- The primary evaluation metric should be realized financial regret, not fuel MAE.
  - Recommended metric: `C_t(g_hat_t) - C_t(g_star_t)`, where `g_star_t` is a cost-optimal feasible reference under the same public-data feasible set.
  - Also report:
    - feasibility and slack use;
    - ramp violations;
    - renewable curtailment;
    - runtime;
    - dispatch distance from observed NYISO fuel mix.

- The model should begin at system-fuel or zone-fuel resolution before generator-level dispatch.
  - System-fuel output dimension: 7.
  - Zone-fuel output dimension: 77.
  - Generator-level output would require stable generator identity, commitment status, ramp data, and private offers that public data does not fully provide.

- Missing constraints must be handled explicitly.
  - If the cost-aware LtO dispatch is much cheaper than observed NYISO dispatch but unrealistic, the likely cause is missing constraints, not discovery of a superior dispatch.
  - We therefore keep dispatch-distance and P-26 commitment-hour diagnostics as guardrails.

- The implemented cost-aware zonal FY model learns zone-fuel marginal proxy costs, not generator-unit offers.
  - Code path: `run_zonal_cost_lto.py`, `ZonalFinancialCostModel`, `train_zonal_financial_cost_model`, and `predict_zonal_financial_cost_lto` in `nyiso_experiments.py`.
  - Decision dimension: 11 NYISO zones times 7 fuel categories.
  - The feasible layer is the existing zonal LP: zonal balances, net injections, internal interface cuts, zonal capacity, weather/availability bounds, and empirical system ramp limits.
  - The training reference is a public-data financial-cost zonal LP oracle using the same feasible set. This follows the feasible-proxy paper's emphasis on integrating feasibility restoration with learning, while keeping our LtO/FY structure from the current Parmentier-style implementation.
  - Source influence:
    - `Literature Review/Energy Dispatch Optimisation Literature/End-to-End Feasible Optimization Proxies for Large-Scale Economic ispatch.pdf`: emphasizes feasible outputs and training through a repair/feasibility layer.
    - `nyiso_experiments.py`: existing Fenchel-Young update uses the difference between the projected oracle and model-induced LP solution as the structured gradient signal.
  - Primary benchmark for this model is financial regret against the cost oracle under the same public-data constraints, not closeness to P-63 alone.
  - Initial test artifact: `data/processed/final_submission/zonal_cost_lto_test_full_eval_e3`.
    - Command: `python3 run_zonal_cost_lto.py --processed-dir data/processed/final_submission --output-dir data/processed/final_submission/zonal_cost_lto_test_full_eval_e3 --max-train-hours 720 --max-eval-hours 0 --structured-epochs 3 --split test`.
    - Test split: 2023-2024, 17,540 evaluated hours.
    - Financial oracle LP mean cost: about 409,004 dollars/hour.
    - Zonal FY LtO mean cost: about 444,798 dollars/hour.
    - Mean modeled financial regret: about 35,794 dollars/hour.
    - Solver success: 100%; mean balance residual near numerical zero; mean interface slack 0 MW.
    - Critical caveat: this artifact was produced before `nyiso_interface_hour.parquet` was present in `data/processed/final_submission`, so the zero interface slack is not evidence of interface-constrained feasibility. Interface-enabled claims now require the P-32 hourly interface table with finite limits.
    - Interpretation: the first implementation is feasible under its loaded public-data constraints and learns a lower-regret policy than the one-epoch smoke run, but still has substantial regret versus the public-data oracle.

- The stronger A1 proxy introduces Gold Book + P-19 capacity processing.
  - Outputs: `data/processed/scuc_proxy/generator_facilities.parquet`, `zonal_fuel_capacity.parquet`, and `zonal_tech_capacity.parquet`.
  - Gold Book capacity is a public fleet snapshot, not historical annual truth.
  - P-19 is used only as a public PTID/name/location join; unmatched Gold Book rows are retained.
  - P-27 masked IDs are not treated as generator identity truth.
  - P-61 is omitted because the local data ends around 2005, outside the 2020-2025 study period.
  - P-54C and P-59 were initially unavailable locally, but later data checks found usable files. They are now processed and tested in A6e/A6f/A8 variants.
  - This keeps the Parmentier-style learned transformation framing honest: `h_theta(x_t)` predicts costs/penalties/derates for an easy public SCUC-inspired LP, not the hidden NYISO SCUC.
  - Initial A0/A1 real-interface ablation, test split 2023-2024, 17,540 evaluated hours, 720 training-hour cap, 3 structured epochs:
    - A0 legacy capacity artifact: `data/processed/final_submission/zonal_cost_lto_ablation_A0_legacy_test_e3`.
    - A0 oracle mean cost: about 410,354 dollars/hour; LtO mean cost: about 463,109 dollars/hour; regret: about 52,755 dollars/hour.
    - A1 Gold Book capacity artifact: `data/processed/final_submission/zonal_cost_lto_ablation_A1_gold_book_test_e3`.
    - A1 oracle mean cost: about 409,866 dollars/hour; LtO mean cost: about 465,540 dollars/hour; regret: about 55,674 dollars/hour.
    - Both runs had 100% solver success, numerical balance residuals, zero interface slack, and enforced non-empty P-32 interface data.
    - Interpretation: Gold Book capacity is the better public fleet input for the SCUC proxy, but the first FY learner did not exploit it without better constraint-stress context.
  - A2/A3 refreshed comparison after adding formal interface diagnostics and enhanced P-511A feature processing:
    - A1/A2 artifact: `data/processed/final_submission/zonal_cost_lto_A2_interface_test_e3`.
    - A1/A2 oracle mean cost: about 409,866 dollars/hour; LtO mean cost: about 445,503 dollars/hour; regret: about 35,637 dollars/hour.
    - A3 artifact: `data/processed/final_submission/zonal_cost_lto_A3_p511a_test_e3`.
    - A3 oracle mean cost: about 409,866 dollars/hour; LtO mean cost: about 449,337 dollars/hour; regret: about 39,470 dollars/hour.
    - A3 improves total fuel-mix MAE, 266.187 MW versus 276.727 MW, but worsens financial regret and overall fuel MAE.
    - Interpretation: P-511A constraint-stress features are useful diagnostics, but the next LtO transformation should learn penalties or derates; simply adding these features to zone-fuel marginal-cost learning is not enough.
  - A4 learned thermal derate test:
    - Artifact: `data/processed/final_submission/zonal_cost_lto_A4_derate_test_e3`.
    - A4 oracle mean cost: about 409,866 dollars/hour; LtO mean cost: about 450,162 dollars/hour; regret: about 40,296 dollars/hour.
    - A4 total fuel-mix MAE improves slightly to 265.745 MW, but financial regret remains worse than A2 and A3.
    - Mean thermal derate is about 0.157 and the maximum derate reaches the 0.500 cap.
    - Interpretation: derates are the right kind of decision variable for public constraint stress, but the first heuristic FY-style derate update is too blunt and should remain experimental.
  - A4r regularized derate test:
    - Artifact: `data/processed/final_submission/zonal_cost_lto_A4r_derate_regularized_test_e3`.
    - Regularization reduced mean thermal derate to about 0.090, but the maximum still reached the 0.500 cap.
    - LtO mean cost rose to about 454,568 dollars/hour; regret rose to about 44,702 dollars/hour.
    - Interpretation: regularization did not rescue the heuristic derate update. The next constraint-aware transformation should use a better target, e.g. supervised/repair-derived derates, or shift to reserve/headroom/outage features rather than continuing this update rule.
  - A5 scheduled outage stress tests:
    - A5a artifact: `data/processed/final_submission/zonal_cost_lto_A5a_outage_test_e3`.
    - A5a adds P-54A scheduled outage stress features to A2. LtO mean cost is about 453,025 dollars/hour; regret is about 43,159 dollars/hour. This is worse than A2.
    - A5b artifact: `data/processed/final_submission/zonal_cost_lto_A5b_outage_p511a_test_e3`.
    - A5b combines P-54A scheduled outage stress with A3 P-511A constraint stress. LtO mean cost falls to about 442,472 dollars/hour; regret falls to about 32,606 dollars/hour.
    - A5b is the best current public SCUC proxy variant on financial regret, but it worsens total fuel-mix MAE to about 286.144 MW. The next ablation should target this tradeoff rather than treating the financial improvement as proof of exact NYISO SCUC reconstruction.
  - A6 ancillary stress and reserve/headroom tests:
    - A6a artifact: `data/processed/final_submission/zonal_cost_lto_A6a_ancillary_test_e3`.
    - A6a adds P-6B ancillary stress features to A5b. It improves total fuel-mix MAE to about 266.318 MW, but worsens LtO mean cost to about 450,359 dollars/hour and regret to about 40,492 dollars/hour.
    - A6b artifact: `data/processed/final_submission/zonal_cost_lto_A6b_headroom_test_e3`.
    - A6b adds a soft ramp-accessible reserve/headroom proxy. It has 0 MW mean reserve slack and improves total fuel-mix MAE to about 269.995 MW, but worsens LtO mean cost to about 445,383 dollars/hour.
    - A6c artifact: `data/processed/final_submission/zonal_cost_lto_A6c_ancillary_headroom_test_e3`.
    - A6c combines P-6B ancillary stress with the reserve/headroom proxy. LtO mean cost falls to about 440,746 dollars/hour; regret is about 32,262 dollars/hour against the reserve-constrained oracle; total fuel-mix MAE improves to about 267.005 MW.
    - A6c reserve-fraction sensitivity artifacts: `zonal_cost_lto_A6c_ancillary_headroom_frac02_test_e3` and `zonal_cost_lto_A6c_ancillary_headroom_frac06_test_e3`.
    - The 2% reserve sensitivity is strongest: LtO mean cost about 440,225 dollars/hour; regret about 30,816 dollars/hour; total fuel-mix MAE about 260.437 MW; 0 MW mean reserve slack.
    - The 6% reserve sensitivity is too restrictive for this proxy: LtO mean cost rises to about 446,549 dollars/hour and regret rises to about 40,248 dollars/hour.
    - Interpretation: A6c at 2% reserve is the best balanced public SCUC proxy candidate so far. It improves A5b's financial dispatch cost and substantially improves the A5b fuel-mix tradeoff, but remains an approximate public proxy, not NYISO SCUC reconstruction.
  - A7a learned reserve penalty test:
    - Artifact: `data/processed/final_submission/zonal_cost_lto_A7a_learned_reserve_penalty_train720_e7`.
    - A7a keeps the A6c 2% reserve/headroom proxy but learns an hourly reserve slack penalty bounded between 5,000 and 200,000.
    - LtO mean cost rises to about 445,487 dollars/hour and regret rises to about 36,078 dollars/hour, worse than A6c 720/e7.
    - Total fuel-mix MAE improves to about 257.676 MW, but the learned penalty reaches the max bound in about 6.7% of hours and reserve slack remains 0 MW.
    - Interpretation: learning only the reserve slack penalty is not the right A7 promotion path. A6c remains the best balanced proxy; a later A7 variant should learn reserve requirement adjustments or use a better supervised/diagnostic target.
  - A7b learned reserve fraction test:
    - Artifact: `data/processed/final_submission/zonal_cost_lto_A7b_learned_reserve_fraction_train720_e7`.
    - A7b keeps the reserve slack penalty fixed and learns a bounded reserve requirement fraction between 0% and 6%.
    - LtO mean cost is about 438,679 dollars/hour and regret is about 29,270 dollars/hour. This is slightly worse financially than A6c 720/e7 but much better than A7a.
    - Total fuel-mix MAE improves to about 253.969 MW, the best reserve-aware fuel-mix diagnostic so far.
    - The learned reserve fraction is not saturated: mean about 1.2%, median about 0.7%, p95 about 4.3%, and 0% of hours at either bound.
    - Interpretation: A7b is a promising reserve-learning tradeoff variant, but A6c remains the best financial/stress-aware candidate. Later A8b supersedes A7b as the strongest pure fuel-mix realism result.
  - A6d/A6e/A6f causal-clean day-ahead feature tests:
    - A6d replaces P-6B real-time ancillary stress with P-5 day-ahead ancillary stress. LtO regret rises to about 44,976 dollars/hour, so it is not promoted over A6c.
    - A6e adds P-54C day-ahead scheduled outage stress. LtO regret rises to about 57,413 dollars/hour, so the first P-54C formulation is not promoted.
    - A6f adds P-59 zonal bid-load features. LtO regret is about 50,875 dollars/hour and total fuel-mix MAE is about 279.768 MW, so the default all-on P-59 path is not promoted.
    - Interpretation: P-5, P-54C, and P-59 are useful public day-ahead evidence, but simply adding them as causal feature families does not guarantee lower public-oracle regret.
  - A8 richer Parmentier-style transformation tests:
    - A8a is the A6f causal-rich feature path with fixed dataset gates at 1.0. LtO regret is about 50,875 dollars/hour, matching the conclusion that the all-on feature path should remain a control.
    - A8b uses P-5 day-ahead ancillary stress to learn reserve requirement fraction and reserve slack penalty. LtO regret is about 48,698 dollars/hour, overall P-63 MAE is about 272.203 MW, and total fuel-mix MAE is about 250.437 MW.
    - A8c adds P-54C-based conservative learned thermal derates to A8b. LtO regret rises to about 53,705 dollars/hour, and the max thermal derate reaches the 0.20 cap.
    - Interpretation: the richer transformation idea is supported diagnostically because A8b improves observed fuel-mix realism, but the current learned reserve/derate objectives do not beat A6c on financial regret. Final reporting should use A6c as the primary financial result and A8b as the secondary fuel-mix realism result.

## Method 2: Out-of-Network Linear Repair Model

- The out-of-network repair model remains the pragmatic baseline.
  - Current pattern: direct neural residual prediction followed by LP repair for balance, capacity, and ramp feasibility. Source: `solve_fuel_repair_dispatch`, `solve_fuel_soft_repair_dispatch`, and `solve_soft_repair_dispatch_many` in `nyiso_experiments.py`.
  - Literature support:
    - repair/projection methods are common for ML OPF proxies. Source: `Machine Learning for Optimal Power Flows.pdf`.
    - E2ELR explicitly combines prediction and feasibility repair layers for ED. Source: `End-to-End Feasible Optimization Proxies for Large-Scale Economic Dispatch.pdf`.

- The repair objective should pivot from pure closeness-to-prediction toward cost-aware repair.
  - Previous repair objective: minimize absolute deviation from raw neural dispatch, with balance penalties.
  - Proposed objective:
    - production-cost proxy;
    - balance/ENS/spill penalties;
    - ramp and availability constraints;
    - small regularization toward predicted or observed dispatch for plausibility.

- The linear repair model should serve as a benchmark against the FY LtO model.
  - It tests whether the extra complexity of learning an in-network objective improves downstream cost and feasibility over "predict then economically repair."
  - It is easier to debug because the cost stack and LP terms are explicit.
  - It is also closer to the repair architectures emphasized in feasible OPF proxy literature.

- The repair model should use the same cost stack as the FY model.
  - This keeps evaluation fair: improvements should come from learning-to-optimize, not from using a different objective definition.
  - Shared cost terms:
    - EIA fuel prices;
    - ATB variable O&M and heat-rate assumptions;
    - EIA-923 validation/heat-rate checks;
    - P-5/P-6B ancillary scarcity features;
    - P-26 operator-commitment features;
    - P-27 bid/UC summary features.

## Data Design Decisions

- Use P-26 as reliability and missing-constraint context.
  - P-26 starts in June 2019, so rows before then must be marked unavailable instead of filled as zero.
  - It contains operator-initiated commitments, not full unit commitment status.
  - Planned features:
    - zone-hour commitment count;
    - committed MW;
    - DARU/OOM counts;
    - constraint-management commitment count.

- Use P-27 as aggregate market-offer context, not unit-specific truth.
  - It spans 2016-01 to 2026-01 for Gen and UC monthly files.
  - It is large and should be streamed/aggregated rather than fully loaded.
  - Planned features:
    - dispatch offer price quantiles;
    - fixed min-gen cost summaries;
    - reserve and regulation offer summaries;
    - startup-cost and min-run/down monthly summaries.

- Use EIA fuel prices as the first production-cost input.
  - Natural gas monthly data is mostly complete for the project window.
  - Petroleum liquids data is sparse and needs missingness flags and cautious filling.
  - Online source: https://www.eia.gov/electricity/data/browser/

- Use EIA-923 to validate and refine cost assumptions.
  - Useful sheets:
    - Page 1 Generation and Fuel Data;
    - Page 4 Generator Data;
    - Page 5 Fuel Receipts and Costs;
    - Page 6 Plant Frame.
  - Use NY state rows and fuel groups such as Natural Gas and Petroleum.
  - 2025 file has a different header offset, so the parser needs header detection.

- Use ATB for technology-level variable O&M and heat-rate assumptions.
  - The local `ATBe.csv` is a 2024 ATB release with projected metric years.
  - For hourly dispatch, use `Variable O&M` and `Heat Rate`; do not use `Fixed O&M` in the dispatch objective.
  - Use `scenario = Moderate` as the default assumption and record sensitivity options later.

- Keep RT price data out of causal day-ahead features.
  - P-24A/P-6B are useful for ex-post realized-cost evaluation and diagnostics.
  - P-2A/P-5 are appropriate causal day-ahead price/scarcity features.
  - Source: `design_descision.md` causal day-ahead feature policy.

## Light Ablation Plan

- Cost-stack ablations should be small and diagnostic.
  - Fixed hand-tuned fuel costs.
  - EIA fuel plus ATB variable O&M.
  - EIA/ATB plus ancillary scarcity features.
  - EIA/ATB plus P-26/P-27 market-context features.

- Model ablations should focus on the two main methods.
  - Out-of-network linear repair:
    - imitation repair objective;
    - cost-aware repair objective.
  - In-network FY LtO:
    - imitation-trained objective;
    - cost/regret-trained objective.

- Weather ablation should be secondary.
  - Weather is already valuable for renewable availability and load context.
  - The main research question is whether cost-aware LtO improves dispatch quality, not whether weather helps.

## Benchmarks and Claims

- Main benchmarks:
  - observed NYISO P-63 dispatch;
  - fixed-cost LP;
  - current imitation LtO;
  - direct neural plus cost-aware repair;
  - cost-aware FY LtO;
  - cost-optimal feasible oracle under the public-data feasible set.

- Main claims we can defend:
  - the LP solutions are feasible under modeled constraints;
  - FY/LtO and repair methods can be compared on realized public-data financial regret;
  - public-data cost proxies improve the realism of the objective compared with pure fuel-mix imitation;
  - zone-fuel dispatch improves spatial fidelity relative to system-fuel dispatch.

- Claims we should not make:
  - true NYISO cost optimality;
  - full UC feasibility;
  - exact private generator production costs;
  - generator-level market reconstruction from masked P-27 data;
  - that observed LBMPs would remain unchanged under counterfactual dispatch.

## Source Index

- Local project sources:
  - `Project Proposal.pdf`
  - `design_descision.md`
  - `nyiso_experiments.py`
  - `nyiso_processing.py`

- Local literature:
  - `Literature Review/Energy Dispatch Optimisation Literature/End-to-End Feasible Optimization Proxies for Large-Scale Economic ispatch.pdf`
  - `Literature Review/Energy Dispatch Optimisation Literature/Learning optimization proxies or large-scale Security Constrained Economic Dispatch.pdf`
  - `Literature Review/Energy Dispatch Optimisation Literature/Machine Learning for Optimal Power Flows.pdf`
  - `Literature Review/Energy Dispatch Optimisation Literature/Predicting AC Optimal Power Flows Combining Deep Learning and Lagrangian Dual Methods.pdf`
  - `Literature Review/Energy Dispatch Optimisation Literature/Task-based End-to-end Model Learning in Stochastic Optimization.pdf`

- Online/public data sources:
  - EIA-923: https://www.eia.gov/electricity/data/eia923/
  - EIA electricity data browser: https://www.eia.gov/electricity/data/browser/
  - EIA heat-rate FAQ: https://www.eia.gov/tools/faqs/faq.php?id=107&t=1
  - NREL Electricity ATB: https://atb.nrel.gov/electricity/
  - NYISO MIS public data portal: https://mis.nyiso.com/public/
