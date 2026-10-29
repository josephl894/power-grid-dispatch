# OUTDATED

This high-level data overview predates the final public SCUC proxy framing. Use
`final_scuc_proxy_results.md`, `ablation_test.md`, `design_descision.md`, and
`research_influence.md` as the active submission documentation.

# NYISO Data Overview for Energy Grid Optimization

## Purpose
This document describes datasets collected from NYISO and related sources to support:
- Energy grid simulation
- Renewable integration modeling
- Reinforcement learning / optimization algorithms
- Hybrid renewable energy system studies

---

## 🧩 Core Data Categories

1. Load & Demand  
2. Generation & Fuel Mix  
3. Transmission & Grid Constraints  
4. Market & Pricing  
5. Asset Metadata  

---

## 🗺️ NYISO Zone Mapping

| Code | Name |
|------|------|
| A | WEST |
| B | GENESE |
| C | CENTRL |
| D | NORTH |
| E | MHK VL |
| F | CAPITL |
| G | HUD VL |
| H | MILLWD |
| I | DUNWOD |
| J | N.Y.C. |
| K | LONGIL |

---

## ⚡ 1. Load & Demand Data

### P-58B — Real-Time Actual Load
- 5-min resolution
- Raw system load
- Used for high-frequency dynamics

### P-58C — Integrated Load
- Hourly (aggregated)
- Primary demand signal for modeling

### P-7 — Load Forecast
- Day-ahead forecast
- Used to model uncertainty

---

## 🌞 2. Generation & Fuel Mix

### P-63 — Real-Time Fuel Mix
Includes:
- Wind
- Hydro
- Nuclear
- Natural Gas
- Other Fossil

Used as ground truth generation.

---

## 🔌 3. Transmission & Constraints

### P-32 — Interface Flows
- Power flow between zones
- Includes limits

### P-33 — Real-Time Constraints
- Binding constraints
- Includes shadow price

### P-511A — Day-Ahead Constraints
- Forecasted congestion

### P-54A / P-54B — Outages
- Scheduled vs real-time outages

---

## 💰 4. Market Data

### P-24A — Real-Time Zonal LBMP
- Real-time electricity price

### P-2A — Day-Ahead LBMP
- Forecasted price

---

## 🏭 5. Asset Metadata

### P-19 — Generator Names
- Generator name
- Zone
- PTID

### Existing Generating Facilities Dataset
Key fields:
- PTID
- fuel_type_1 / fuel_type_2
- unit_type
- Dual flag

---

## 🔑 Fuel Type Codes

| Code | Meaning |
|------|--------|
| BAT | Battery |
| BUT | Butane |
| FO2 | No. 2 Fuel Oil |
| FO4 | No. 4 Fuel Oil |
| FO6 | No. 6 Fuel Oil |
| FW | Flywheel |
| JF | Jet Fuel |
| KER | Kerosene |
| MTE | Methane (Biogas) |
| NG | Natural Gas |
| OT | Other |
| REF | Refuse |
| SUN | Sunlight |
| UR | Uranium |
| WAT | Water |
| WD | Wood |
| LBW | Land-Based Wind |
| OSW | Offshore Wind |

---

## ⚙️ Unit Type Codes

| Code | Meaning |
|------|--------|
| CC | Combined Cycle |
| CG | Cogeneration |
| CT | Combustion Turbine (CC portion) |
| CW | Waste Heat Only |
| ES | Energy Storage |
| FC | Fuel Cell |
| GT | Gas Turbine |
| HY | Conventional Hydro |
| IC | Internal Combustion |
| JE | Jet Engine |
| NB | Nuclear (BWR) |
| NP | Nuclear (PWR) |
| PS | Pumped Storage |
| PV | Photovoltaic |
| ST | Steam Turbine |
| WT | Wind Turbine |

---

## 🗺️ New York County Codes

State: NY (36)

Example subset:
- 001: Albany
- 029: Erie
- 047: Kings (Brooklyn)
- 061: New York (Manhattan)
- 081: Queens
- 085: Richmond (Staten Island)
- 103: Suffolk
- 119: Westchester

(Full mapping included in source dataset)

---

## 🏷️ Final Fuel Classification

Mapped to:

- Dual
- Natural Gas
- Nuclear
- Wind
- Hydro
- Other Fossil Fuels
- Other Renewables

Final dataset structure:
---

## 🌍 External Data (NASA POWER)

Used for:
- Solar irradiance (GHI)
- Wind speed (50m)
- Wind direction
- Temperature
- Precipitation

Purpose:
- Map weather → renewable generation

Hourly NASA POWER weather fields:

| Column | Meaning | Unit |
|--------|---------|------|
| ALLSKY_SFC_SW_DWN | All-sky surface shortwave downward irradiance, used as GHI | Wh/m^2 |
| WS50M | Wind speed at 50 meters | m/s |
| WD50M | Wind direction at 50 meters | Degrees |
| T2M | Temperature at 2 meters | C |
| PRECTOT / PRECTOTCORR | Corrected precipitation | mm/day in API metadata |

Precipitation note:
- NASA POWER currently returns `PRECTOT` requests as `PRECTOTCORR` in hourly JSON/CSV output.
- If a `PRECTOT` column is empty, check for `PRECTOTCORR`.
- For hourly energy-modeling features, convert the returned precipitation rate to hourly depth with:
  `precip_mm_this_hour = PRECTOTCORR / 24`
- NASA hourly timestamps should be kept in UTC.

---

## 🔄 Data Relationships

- Load (P-58) + Generation (P-63) → system balance
- P-32 + P-33 → network constraints
- LBMP → economic signals
- PTID → primary join key

---

## ⚠️ Known Limitations

- BTM solar not fully captured
- Some PTIDs missing → default assumptions
- Spatial resolution limited to zones
- Generator ramp constraints not fully available

---

## 🚀 Intended Use

Supports:
- RL-based dispatch optimization
- Grid simulation
- Renewable integration modeling
- Synthetic data generation

---

## 📌 Notes for Future Models

- Always align timestamps
- Prefer P-58C for stability
- Use zonal abstraction unless modeling full network
- PTID is the primary key for all joins

---

## 🧪 Current Implemented Modeling Pipeline

The active implementation no longer uses iHOGA. It builds a NYISO
weather-aware system-level fuel dispatch imitation benchmark.

### Processed outputs

Generated by `nyiso_processing.py`:

- `data/processed/nyiso_weather_zone_hour.parquet`
  - one row per `(timestamp_utc, zone_code)`
  - weather fields: `ghi_wh_m2`, `wind_speed_50m_mps`,
    `wind_dir_50m_deg`, `temp_2m_c`
  - engineered fields: `wind_dir_sin`, `wind_dir_cos`,
    `wind_speed_cubed`, `ghi_rolling_3h`, `temp_rolling_24h`
- `data/processed/nyiso_zone_hour.parquet`
  - one row per `(timestamp_utc, zone_code)`
  - joins actual load, day-ahead forecast load, DA/RT LBMP, congestion,
    and zone weather
- `data/processed/nyiso_system_hour.parquet`
  - one row per `timestamp_utc`
  - system load, forecast error, market/constraint/outage features,
    capacity-weighted weather aggregates, static capacity features,
    and P-63 fuel-mix oracle target columns

### Oracle action

The first modeled action is system-level fuel dispatch:

```text
a_t = [
  Dual Fuel,
  Natural Gas,
  Hydro,
  Nuclear,
  Wind,
  Other Renewables,
  Other Fossil Fuels
]
```

Target columns:

```text
gen_dual_fuel_mw
gen_natural_gas_mw
gen_hydro_mw
gen_nuclear_mw
gen_wind_mw
gen_other_renewables_mw
gen_other_fossil_fuels_mw
```

### Easy dispatch LP

For each hour, learned or fixed fuel costs drive the deterministic LP:

```text
min sum_f c_f,t * g_f,t
    + residual_penalty * residual_supply_t
    + ens_penalty * ens_t
    + spill_penalty * spill_t
```

subject to:

```text
sum_f g_f,t + residual_supply_t + ens_t = load_t + spill_t
0 <= g_f,t <= available_capacity_f
```

Available capacity is:

```text
max(static_capacity_f, 1.05 * train_max_observed_generation_f)
```

Network constraints from P-32/P-33/P-511A are included as features in v1,
not as hard DC power-flow constraints.
