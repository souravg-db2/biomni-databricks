-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Register Biomni Biochemistry Tools as Unity Catalog Functions
-- MAGIC
-- MAGIC This notebook registers biochemistry analysis functions from the
-- MAGIC [Biomni](https://github.com/snap-stanford/Biomni) project as Unity Catalog (UC) Python functions
-- MAGIC under the `biomni.agent` schema so they can be used as tools by Databricks agents.
-- MAGIC
-- MAGIC **Functions registered:**
-- MAGIC 1. `analyze_circular_dichroism_spectra` — CD spectroscopy analysis
-- MAGIC 2. `analyze_rna_secondary_structure_features` — RNA structure feature extraction
-- MAGIC 3. `analyze_protease_kinetics` — Protease kinetics (Michaelis-Menten fitting)
-- MAGIC 4. `analyze_enzyme_kinetics_assay` — In vitro enzyme kinetics assay simulation and analysis
-- MAGIC 5. `analyze_itc_binding_thermodynamics` — ITC binding affinity analysis
-- MAGIC 6. `analyze_protein_conservation` — Protein sequence conservation analysis

-- COMMAND ----------

USE CATALOG biomni;
USE SCHEMA agent;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1. Analyze Circular Dichroism Spectra
-- MAGIC Analyzes CD spectroscopy data for secondary structure determination and thermal stability.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION biomni.agent.analyze_circular_dichroism_spectra(
    sample_name STRING
        COMMENT 'Name of the biomolecule sample (e.g. "Znf706", "G-quadruplex")',
    sample_type STRING
        COMMENT 'Type of biomolecule: "protein" or "nucleic_acid"',
    wavelength_data ARRAY<DOUBLE>
        COMMENT 'Wavelength values in nm for CD spectrum',
    cd_signal_data ARRAY<DOUBLE>
        COMMENT 'CD signal intensity values (typically in mdeg or delta-epsilon)',
    temperature_data ARRAY<DOUBLE>
        COMMENT 'Temperature values in Celsius for thermal denaturation. Pass empty array if not available.',
    thermal_cd_data ARRAY<DOUBLE>
        COMMENT 'CD signal values at a specific wavelength across different temperatures. Pass empty array if not available.'
)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Analyzes circular dichroism (CD) spectroscopy data to determine secondary structure and thermal stability of proteins or nucleic acids. Returns a structured research log summarizing secondary structure classification, thermal melting temperature (Tm), and cooperativity of the unfolding transition.'
AS $$
    import numpy as np
    from datetime import datetime

    log = f"# Circular Dichroism Analysis Report for {sample_name}\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    log += "## Sample Information\n"
    log += f"- Sample Name: {sample_name}\n"
    log += f"- Sample Type: {sample_type}\n\n"

    wl = np.array(wavelength_data)
    cd = np.array(cd_signal_data)

    if len(wl) != len(cd):
        return "Error: wavelength_data and cd_signal_data must have the same length."

    log += "## Secondary Structure Analysis\n"

    structure = "undetermined"

    if sample_type.lower() == "protein":
        alpha_helix_signal = np.sum((wl >= 190) & (wl <= 195) & (cd > 0))
        beta_sheet_signal = np.sum((wl >= 215) & (wl <= 220) & (cd < 0))
        random_coil_signal = np.sum((wl >= 195) & (wl <= 200) & (cd < 0))

        if alpha_helix_signal > beta_sheet_signal and alpha_helix_signal > random_coil_signal:
            structure = "predominantly alpha-helical"
        elif beta_sheet_signal > alpha_helix_signal and beta_sheet_signal > random_coil_signal:
            structure = "predominantly beta-sheet"
        else:
            structure = "mixed or predominantly random coil"

        log += f"- The CD spectrum indicates {structure} structure for {sample_name}.\n"
        log += "- Key spectral features:\n"
        log += "  - 190-195 nm region: associated with alpha-helical content\n"
        log += "  - 215-220 nm region: associated with beta-sheet content\n\n"

    elif sample_type.lower() == "nucleic_acid":
        g_quad = np.sum((wl >= 290) & (wl <= 300) & (cd > 0))
        b_form = np.sum((wl >= 270) & (wl <= 280) & (cd > 0))

        if g_quad > 0:
            structure = "G-quadruplex characteristics"
        elif b_form > 0:
            structure = "B-form characteristics"
        else:
            structure = "non-standard structure"

        log += f"- The CD spectrum indicates {structure} for {sample_name}.\n"
        log += "- Key spectral features:\n"
        log += "  - 290-300 nm positive peak: characteristic of G-quadruplex structures\n"
        log += "  - 270-280 nm positive peak: characteristic of B-form DNA\n\n"
    else:
        log += f"- Unknown sample_type '{sample_type}'. Expected 'protein' or 'nucleic_acid'.\n\n"

    has_thermal = (
        temperature_data is not None and len(temperature_data) > 0
        and thermal_cd_data is not None and len(thermal_cd_data) > 0
    )
    tm = None
    cooperativity = None

    if has_thermal:
        t_arr = np.array(temperature_data)
        th_cd = np.array(thermal_cd_data)

        if len(t_arr) != len(th_cd):
            log += "Warning: temperature_data and thermal_cd_data have different lengths. Skipping thermal analysis.\n\n"
        else:
            log += "## Thermal Stability Analysis\n"

            min_sig = np.min(th_cd)
            max_sig = np.max(th_cd)
            if max_sig - min_sig == 0:
                log += "- Warning: No variation in thermal CD signal. Cannot determine Tm.\n\n"
            else:
                unfolded_fraction = (th_cd - min_sig) / (max_sig - min_sig)
                tm_idx = np.argmin(np.abs(unfolded_fraction - 0.5))
                tm = t_arr[tm_idx]

                log += f"- Estimated melting temperature (Tm): {tm:.1f} deg C\n"

                t_range = t_arr[-1] - t_arr[0]
                transition_width = (
                    t_range / len(t_arr)
                    * np.sum((unfolded_fraction > 0.2) & (unfolded_fraction < 0.8))
                )

                if transition_width < 0.2 * t_range:
                    cooperativity = "highly cooperative (sharp transition)"
                elif transition_width < 0.4 * t_range:
                    cooperativity = "moderately cooperative"
                else:
                    cooperativity = "non-cooperative (broad transition)"

                log += f"- Thermal transition: {cooperativity}\n\n"

    log += "## Conclusions\n"
    if sample_type.lower() == "protein":
        log += f"- {sample_name} shows {structure} according to CD spectroscopy.\n"
    else:
        log += f"- {sample_name} exhibits {structure} according to CD spectroscopy.\n"

    if tm is not None and cooperativity is not None:
        log += f"- The molecule has a melting temperature of {tm:.1f} deg C with {cooperativity}.\n"

    return log
$$;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. Analyze RNA Secondary Structure Features
-- MAGIC Calculates structural features (stems, loops, base pairs, energy) from dot-bracket notation.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION biomni.agent.analyze_rna_secondary_structure_features(
    dot_bracket_structure STRING
        COMMENT 'RNA secondary structure in dot-bracket notation. Parentheses represent base pairs, dots represent unpaired bases. Example: "(((...)))"',
    sequence STRING
        COMMENT 'The RNA sequence corresponding to the structure. Pass empty string if not available. If provided, sequence-dependent energy calculations will be performed.'
)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Calculates numeric values for various structural features of an RNA secondary structure given dot-bracket notation. Returns a research log with total base pairs, stem counts and details, loop sizes, paired/unpaired base statistics, and optional free energy estimates when a sequence is provided.'
AS $$
    log = "# RNA Secondary Structure Feature Analysis\n\n"

    if not all(c in "().[]{}" for c in dot_bracket_structure):
        return "Error: Invalid dot-bracket notation. Use only '()', '[]', '{}', and '.'."

    log += f"Input structure (length: {len(dot_bracket_structure)}): {dot_bracket_structure}\n"

    seq = sequence if sequence and len(sequence) > 0 else None
    if seq:
        log += f"Input sequence (length: {len(seq)}): {seq}\n"
        if len(seq) != len(dot_bracket_structure):
            return "Error: Sequence and structure lengths do not match."

    pairs = []
    stack = []
    for i, char in enumerate(dot_bracket_structure):
        if char in "([{":
            stack.append((i, char))
        elif char in ")]}":
            if not stack:
                return "Error: Unbalanced structure. More closing than opening brackets."
            j, opening = stack.pop()
            matching = {"(": ")", "[": "]", "{": "}"}
            if matching[opening] != char:
                return "Error: Mismatched bracket types."
            pairs.append((j, i))

    if stack:
        return "Error: Unbalanced structure. More opening than closing brackets."

    pairs.sort()

    stems = []
    current_stem = []
    for i, (start, end) in enumerate(pairs):
        if current_stem and (start != pairs[i - 1][0] + 1 or end != pairs[i - 1][1] - 1):
            stems.append(current_stem)
            current_stem = []
        current_stem.append((start, end))
    if current_stem:
        stems.append(current_stem)

    stem_lengths = [len(s) for s in stems]

    loops = []
    for i in range(len(stems)):
        stem = stems[i]
        last_pair = stem[-1]
        next_start = stems[i + 1][0][0] if i < len(stems) - 1 else len(dot_bracket_structure)
        loop_size = next_start - last_pair[1] - 1
        if loop_size > 0:
            loops.append(loop_size)

    total_paired = len(pairs) * 2
    total_unpaired = len(dot_bracket_structure) - total_paired

    stem_energies = []
    if seq and len(stems) > 0:
        energy_params = {
            "AU": -0.9, "UA": -0.9,
            "GC": -2.1, "CG": -2.1,
            "GU": -0.5, "UG": -0.5,
        }
        for stem in stems:
            e = 0.0
            for s, en in stem:
                if s < len(seq) and en < len(seq):
                    e += energy_params.get(seq[s] + seq[en], 0.0)
            stem_energies.append(e)

    n = len(dot_bracket_structure)
    log += "\n## Structural Features\n\n"
    log += f"Total base pairs: {len(pairs)}\n"
    log += f"Number of stems: {len(stems)}\n"
    log += f"Longest stem length: {max(stem_lengths) if stem_lengths else 0}\n"
    log += f"Average stem length: {sum(stem_lengths) / len(stem_lengths) if stem_lengths else 0:.2f}\n"
    log += f"Paired bases: {total_paired} ({total_paired / n * 100:.1f}%)\n"
    log += f"Unpaired bases: {total_unpaired} ({total_unpaired / n * 100:.1f}%)\n"

    if loops:
        log += f"Number of loops: {len(loops)}\n"
        log += f"Average loop size: {sum(loops) / len(loops):.2f}\n"
        log += f"Largest loop size: {max(loops)}\n"

    if seq and stem_energies:
        log += "\n## Energy Calculations\n\n"
        log += f"Total estimated free energy: {sum(stem_energies):.2f} kcal/mol\n"
        if len(stems) >= 2:
            log += f"Upstream stem free energy: {stem_energies[0]:.2f} kcal/mol\n"
            log += f"Downstream stem free energy: {stem_energies[-1]:.2f} kcal/mol\n"
        if stem_lengths and stem_lengths[0] >= 3:
            log += f"Zipper stem free energy: {stem_energies[0]:.2f} kcal/mol\n"

    log += "\n## Stem Details\n\n"
    for i, stem in enumerate(stems):
        log += f"Stem {i + 1}: {len(stem)} base pairs\n"
        log += f"  Positions: {stem[0][0]}-{stem[0][1]} to {stem[-1][0]}-{stem[-1][1]}\n"
        if seq and i < len(stem_energies):
            log += f"  Estimated stability: {stem_energies[i]:.2f} kcal/mol\n"

    return log
$$;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 3. Analyze Protease Kinetics
-- MAGIC Fits Michaelis-Menten kinetics to fluorogenic peptide cleavage assay data.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION biomni.agent.analyze_protease_kinetics(
    time_points ARRAY<DOUBLE>
        COMMENT 'Array of time points in seconds at which fluorescence was measured',
    fluorescence_data_json STRING
        COMMENT 'JSON-encoded 2D array of fluorescence measurements. Each inner array corresponds to one substrate concentration and contains fluorescence values at each time point. Example: "[[1.0, 2.0, 3.0], [1.5, 3.0, 4.5]]"',
    substrate_concentrations ARRAY<DOUBLE>
        COMMENT 'Array of substrate concentrations in micromolar (uM), one per row in fluorescence_data',
    enzyme_concentration DOUBLE
        COMMENT 'Concentration of the protease enzyme in micromolar (uM)'
)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Analyzes protease kinetics data from fluorogenic peptide cleavage assays. Calculates initial velocities, fits to the Michaelis-Menten equation, and determines Vmax, KM, kcat, and catalytic efficiency (kcat/KM).'
AS $$
    import json
    import numpy as np
    from scipy.optimize import curve_fit

    time_pts = np.array(time_points)
    fluor_data = np.array(json.loads(fluorescence_data_json))
    sub_concs = np.array(substrate_concentrations)

    if fluor_data.shape[0] != len(sub_concs):
        return "Error: Number of rows in fluorescence_data_json must equal length of substrate_concentrations."
    if fluor_data.shape[1] != len(time_pts):
        return "Error: Number of columns in fluorescence_data_json must equal length of time_points."

    initial_velocities = np.zeros(len(sub_concs))
    for i in range(len(sub_concs)):
        num_points = max(5, int(len(time_pts) * 0.2))
        num_points = min(num_points, len(time_pts))
        slope, _ = np.polyfit(time_pts[:num_points], fluor_data[i, :num_points], 1)
        initial_velocities[i] = slope

    def michaelis_menten(s, vmax, km):
        return vmax * s / (km + s)

    try:
        params, covariance = curve_fit(
            michaelis_menten, sub_concs, initial_velocities,
            p0=[max(initial_velocities), np.mean(sub_concs)],
            bounds=([0, 0], [np.inf, np.inf]),
        )
        vmax, km = params
        std_dev = np.sqrt(np.diag(covariance))
        vmax_std, km_std = std_dev

        kcat = vmax / enzyme_concentration
        kcat_std = vmax_std / enzyme_concentration
        cat_eff = kcat / km
        cat_eff_std = cat_eff * np.sqrt((kcat_std / kcat) ** 2 + (km_std / km) ** 2)

        log = "# Protease Kinetics Analysis Report\n\n"
        log += f"## Experimental Setup\n"
        log += f"- Substrate concentrations tested: {len(sub_concs)}\n"
        log += f"- Time points: {len(time_pts)}\n"
        log += f"- Enzyme concentration: {enzyme_concentration} uM\n\n"
        log += "## Initial Velocities\n"
        for s, v in zip(sub_concs, initial_velocities):
            log += f"- [{s:.2f} uM]: v0 = {v:.4f} a.u./s\n"
        log += "\n## Michaelis-Menten Fit Results\n"
        log += f"- Vmax: {vmax:.4f} +/- {vmax_std:.4f} a.u./s\n"
        log += f"- KM: {km:.4f} +/- {km_std:.4f} uM\n"
        log += f"- kcat: {kcat:.4f} +/- {kcat_std:.4f} s^-1\n"
        log += f"- Catalytic efficiency (kcat/KM): {cat_eff:.4f} +/- {cat_eff_std:.4f} uM^-1 s^-1\n"

        return log

    except Exception as e:
        return f"Error during Michaelis-Menten fitting: {str(e)}"
$$;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4. Analyze Enzyme Kinetics Assay
-- MAGIC Simulates and analyzes an in vitro enzyme kinetics assay with optional modulator dose-response.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION biomni.agent.analyze_enzyme_kinetics_assay(
    enzyme_name STRING
        COMMENT 'Name of the purified enzyme being tested',
    substrate_concentrations ARRAY<DOUBLE>
        COMMENT 'Array of substrate concentrations in micromolar (uM) for kinetic analysis',
    enzyme_concentration DOUBLE
        COMMENT 'Concentration of the enzyme in nanomolar (nM)',
    modulators_json STRING
        COMMENT 'JSON object of modulators. Keys are modulator names, values are arrays of concentrations in uM. Pass empty string or "{}" if no modulators. Example: {"InhibitorA": [0, 1, 5, 10, 25, 50, 100]}',
    time_points ARRAY<DOUBLE>
        COMMENT 'Time points in minutes for time-course measurements. Pass empty array to use defaults [0, 5, 10, 15, 20, 30, 45, 60].'
)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Performs in vitro enzyme kinetics assay analysis including time-course determination of linear range, Michaelis-Menten kinetic parameter fitting (Vmax, Km), and dose-dependent modulator effect analysis with IC50 estimation.'
AS $$
    import json
    import numpy as np
    from scipy.optimize import curve_fit

    if time_points is None or len(time_points) == 0:
        t_pts = np.array([0, 5, 10, 15, 20, 30, 45, 60])
    else:
        t_pts = np.array(time_points)

    modulators = {}
    if modulators_json and modulators_json.strip() and modulators_json.strip() != "{}":
        modulators = json.loads(modulators_json)

    log = f"## In Vitro Enzyme Kinetics Assay: {enzyme_name}\n\n"
    log += f"Enzyme concentration: {enzyme_concentration} nM\n"

    def michaelis_menten(s, vmax, km):
        return vmax * s / (km + s)

    # 1. Time-course kinetic assay (simulated)
    log += "\n### Time-Course Kinetic Assay\n\n"
    max_activity = 100.0
    rate_constant = 0.05

    np.random.seed(42)
    time_course_activity = max_activity * (1 - np.exp(-rate_constant * t_pts))
    time_course_activity += np.random.normal(0, 3, len(t_pts))

    linear_indices = np.where(time_course_activity >= 0.3 * max_activity)[0]
    if len(linear_indices) > 0:
        linear_cutoff = t_pts[linear_indices[0]]
    else:
        linear_cutoff = t_pts[-1]

    log += f"Linear range determined to be 0-{linear_cutoff:.0f} minutes.\n"
    log += "\nTime-course data:\n"
    for t, a in zip(t_pts, time_course_activity):
        log += f"  {t:.0f} min: {a:.2f} units\n"

    # 2. Substrate kinetics (Michaelis-Menten)
    log += "\n### Substrate Kinetics Analysis\n\n"

    true_vmax = 120.0
    true_km = 25.0
    sub_arr = np.array(substrate_concentrations)

    activity_values = michaelis_menten(sub_arr, true_vmax, true_km)
    activity_values += np.random.normal(0, 5, len(sub_arr))

    vmax, km = true_vmax, true_km
    try:
        params, _ = curve_fit(
            michaelis_menten, sub_arr, activity_values,
            p0=[100, 20], bounds=([0, 0], [500, 200]),
        )
        vmax, km = params

        log += "Michaelis-Menten parameters:\n"
        log += f"- Vmax: {vmax:.2f} units\n"
        log += f"- Km: {km:.2f} uM\n\n"
        log += "Substrate kinetics data:\n"
        for s, a in zip(sub_arr, activity_values):
            log += f"  [{s:.1f} uM]: {a:.2f} units\n"
    except Exception:
        log += "Error: Could not fit data to Michaelis-Menten model.\n"

    # 3. Modulator effects
    if modulators:
        log += "\n### Modulator Effects Analysis\n\n"

        for mod_name, concentrations in modulators.items():
            log += f"#### Modulator: {mod_name}\n\n"

            ic50 = np.random.uniform(1, 50)
            hill_coef = 1.0

            mod_activities = []
            for conc in concentrations:
                if conc == 0:
                    act = 100.0
                else:
                    act = 100.0 / (1 + (conc / ic50) ** hill_coef)
                act += np.random.normal(0, 3)
                mod_activities.append(act)

            log += "Dose-response data:\n"
            for conc, act in zip(concentrations, mod_activities):
                log += f"  [{conc:.1f} uM]: {act:.2f}% activity\n"

            nonzero_conc = np.array([c for c in concentrations if c > 0])
            nonzero_act = np.array([a for c, a in zip(concentrations, mod_activities) if c > 0])

            if len(nonzero_conc) >= 3:
                try:
                    def dose_response(x, ic50_fit, hill):
                        return 100.0 / (1 + (x / ic50_fit) ** hill)

                    p, _ = curve_fit(
                        dose_response, nonzero_conc, nonzero_act,
                        p0=[10, 1], bounds=([0.1, 0.1], [1000, 10]),
                    )
                    log += f"\nDose-response fit:\n"
                    log += f"- IC50: {p[0]:.2f} uM\n"
                    log += f"- Hill coefficient: {p[1]:.2f}\n\n"
                except Exception:
                    log += f"\nCould not fit dose-response curve for {mod_name}.\n\n"
            else:
                log += f"\nInsufficient data points to calculate IC50 for {mod_name}.\n\n"

    # Summary
    log += "\n### Summary\n\n"
    log += f"Completed in vitro enzyme kinetics assay for {enzyme_name}.\n"
    log += f"- Linear range: 0-{linear_cutoff:.0f} minutes\n"
    log += f"- Vmax: {vmax:.2f} units, Km: {km:.2f} uM\n"
    if modulators:
        log += f"- Modulators tested: {', '.join(modulators.keys())}\n"

    return log
$$;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 5. Analyze ITC Binding Thermodynamics
-- MAGIC Fits isothermal titration calorimetry data to a one-site binding model.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION biomni.agent.analyze_itc_binding_thermodynamics(
    injection_numbers ARRAY<DOUBLE>
        COMMENT 'Array of injection numbers (1, 2, 3, ...)',
    injection_volumes ARRAY<DOUBLE>
        COMMENT 'Array of injection volumes in microliters',
    heats ARRAY<DOUBLE>
        COMMENT 'Array of heat released/absorbed per injection in microcalories',
    temperature DOUBLE
        COMMENT 'Temperature in Kelvin at which experiment was conducted (e.g. 298.15 for 25 deg C)',
    protein_concentration DOUBLE
        COMMENT 'Initial concentration of protein in the cell in molar (M). Use 0 for unknown.',
    ligand_concentration DOUBLE
        COMMENT 'Concentration of ligand in the syringe in molar (M). Use 0 for unknown.'
)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Analyzes isothermal titration calorimetry (ITC) data to determine binding affinity and thermodynamic parameters. Fits a one-site binding model and returns dissociation constant (Kd), binding enthalpy (delta-H), entropy (delta-S), Gibbs free energy (delta-G), and stoichiometry (n).'
AS $$
    import numpy as np
    from scipy.optimize import curve_fit

    log = "# ITC Binding Affinity Analysis\n\n"

    inj = np.array(injection_numbers)
    vols = np.array(injection_volumes)
    heat_vals = np.array(heats)

    if not (len(inj) == len(vols) == len(heat_vals)):
        return "Error: injection_numbers, injection_volumes, and heats must all have the same length."

    log += f"## Data Summary\n"
    log += f"- Number of injections: {len(inj)}\n"
    log += f"- Temperature: {temperature} K ({temperature - 273.15:.1f} deg C)\n"

    prot_conc = protein_concentration if protein_concentration > 0 else 1.0
    lig_conc = ligand_concentration if ligand_concentration > 0 else 10.0

    if protein_concentration <= 0 or ligand_concentration <= 0:
        log += "- Warning: Protein or ligand concentration not provided. Using normalized values.\n"

    log += f"- Protein concentration: {prot_conc:.2e} M\n"
    log += f"- Ligand concentration: {lig_conc:.2e} M\n\n"

    cell_volume = 1.4  # mL

    cumulative_volume = np.cumsum(vols)
    dilution_factor = 1 - (cumulative_volume / cell_volume)
    prot_corrected = prot_conc * dilution_factor

    molar_ratios = np.zeros(len(inj))
    for i in range(len(inj)):
        lig_added = vols[i] * lig_conc / 1000.0
        if i == 0:
            molar_ratios[i] = lig_added / (prot_conc * cell_volume / 1000.0)
        else:
            denom = prot_corrected[i] * (cell_volume - cumulative_volume[i]) / 1000.0
            if denom > 0:
                molar_ratios[i] = molar_ratios[i - 1] + lig_added / denom
            else:
                molar_ratios[i] = molar_ratios[i - 1]

    log += "## Model Fitting\n"
    log += "- Applying one-site binding model\n\n"

    def one_site_model(x, Kd, dH, n):
        Ka = 1.0 / Kd
        q = np.zeros_like(x)
        for i in range(len(x)):
            bound = (n * prot_corrected[min(i, len(prot_corrected)-1)] * Ka * (x[i] * prot_corrected[min(i, len(prot_corrected)-1)])) / (1 + Ka * (x[i] * prot_corrected[min(i, len(prot_corrected)-1)]))
            if i == 0:
                q[i] = bound * dH * cell_volume
            else:
                bound_prev = (n * prot_corrected[min(i-1, len(prot_corrected)-1)] * Ka * (x[i-1] * prot_corrected[min(i-1, len(prot_corrected)-1)])) / (1 + Ka * (x[i-1] * prot_corrected[min(i-1, len(prot_corrected)-1)]))
                q[i] = bound * dH * (cell_volume - cumulative_volume[min(i, len(cumulative_volume)-1)]) - bound_prev * dH * (cell_volume - cumulative_volume[min(i-1, len(cumulative_volume)-1)])
        return q

    try:
        popt, pcov = curve_fit(
            one_site_model, molar_ratios, heat_vals,
            p0=[1e-6, -5000.0, 1.0], maxfev=10000,
        )
        Kd, dH, n = popt
        perr = np.sqrt(np.diag(pcov))
        Kd_err, dH_err, n_err = perr

        R = 1.9872  # cal/(mol*K)
        dG = R * temperature * np.log(Kd)
        dS = (dH - dG) / temperature

        residuals = heat_vals - one_site_model(molar_ratios, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((heat_vals - np.mean(heat_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        log += "## Results\n\n"
        log += f"- Binding Stoichiometry (n): {n:.2f} +/- {n_err:.2f}\n"
        log += f"- Dissociation Constant (Kd): {Kd * 1e6:.2f} +/- {Kd_err * 1e6:.2f} uM\n"
        log += f"- Association Constant (Ka): {1 / Kd / 1e6:.2f} x 10^6 M^-1\n"
        log += f"- Binding Enthalpy (dH): {dH:.2f} +/- {dH_err:.2f} cal/mol\n"
        log += f"- Binding Entropy (dS): {dS:.2f} cal/(mol*K)\n"
        log += f"- Gibbs Free Energy (dG): {dG:.2f} cal/mol\n"
        log += f"- R-squared: {r_squared:.4f}\n"

    except Exception as e:
        log += f"- Error during model fitting: {str(e)}\n"
        log += "- Consider trying different initial parameter guesses or a different binding model.\n"

    log += "\n## Conclusion\n"
    log += "Analysis complete using a one-site binding model.\n"

    return log
$$;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 6. Analyze Protein Conservation
-- MAGIC Pure-Python protein sequence conservation analysis (no Biopython dependency).

-- COMMAND ----------

CREATE OR REPLACE FUNCTION biomni.agent.analyze_protein_conservation(
    sequences_json STRING
        COMMENT 'JSON array of objects with "id" and "sequence" keys. Example: [{"id": "Human_TP53", "sequence": "MEEPQ..."}, {"id": "Mouse_Tp53", "sequence": "MEESQ..."}]'
)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Performs protein sequence conservation analysis across multiple organisms. Computes a simple pairwise identity-based distance matrix, identifies highly conserved positions (>80% identity), and returns a detailed research log. Input sequences should be pre-aligned (same length, with gaps as "-") for best results, or will be padded to equal length.'
AS $$
    import json

    log = "# Protein Sequence Conservation Analysis\n\n"

    try:
        entries = json.loads(sequences_json)
    except json.JSONDecodeError as e:
        return f"Error: Could not parse sequences_json: {str(e)}"

    if not isinstance(entries, list) or len(entries) < 2:
        return "Error: sequences_json must be a JSON array with at least 2 sequence entries."

    ids = []
    seqs = []
    for entry in entries:
        if "id" not in entry or "sequence" not in entry:
            return "Error: Each entry must have 'id' and 'sequence' keys."
        ids.append(entry["id"])
        seqs.append(entry["sequence"].upper().replace(" ", ""))

    log += f"## Input\n"
    log += f"- Number of sequences: {len(ids)}\n"
    for sid, seq in zip(ids, seqs):
        log += f"  - {sid}: {len(seq)} residues\n"

    max_len = max(len(s) for s in seqs)
    aligned = [s.ljust(max_len, "-") for s in seqs]

    lengths_differ = len(set(len(s) for s in seqs)) > 1
    if lengths_differ:
        log += "\n- Note: Sequences have different lengths. Shorter sequences were padded with gaps for alignment.\n"

    log += f"\n## Alignment\n"
    log += f"- Alignment length: {max_len} positions\n\n"

    # Conservation analysis
    conserved_positions = []
    consensus = []
    conservation_scores = []

    for pos in range(max_len):
        column = [s[pos] for s in aligned]
        non_gap = [c for c in column if c != "-"]

        if not non_gap:
            consensus.append("-")
            conservation_scores.append(0.0)
            continue

        counts = {}
        for c in column:
            counts[c] = counts.get(c, 0) + 1

        most_common = max(non_gap, key=lambda c: counts.get(c, 0))
        score = counts.get(most_common, 0) / len(column)

        consensus.append(most_common)
        conservation_scores.append(score)

        if score > 0.8:
            conserved_positions.append(pos + 1)

    log += "## Conservation Results\n\n"
    log += f"- Total positions: {max_len}\n"
    log += f"- Highly conserved positions (>80%): {len(conserved_positions)}\n"
    log += f"- Conservation rate: {len(conserved_positions) / max_len * 100:.1f}%\n"

    if conserved_positions:
        shown = conserved_positions[:20]
        log += f"- Conserved positions: {', '.join(str(p) for p in shown)}"
        if len(conserved_positions) > 20:
            log += f" ... and {len(conserved_positions) - 20} more"
        log += "\n"

    fully_conserved = [p for p, s in zip(range(1, max_len + 1), conservation_scores) if s == 1.0]
    log += f"- Fully conserved (100%): {len(fully_conserved)} positions\n"

    # Pairwise identity matrix
    log += "\n## Pairwise Sequence Identity\n\n"
    n_seq = len(aligned)
    for i in range(n_seq):
        for j in range(i + 1, n_seq):
            matches = sum(1 for a, b in zip(aligned[i], aligned[j]) if a == b and a != "-")
            comparable = sum(1 for a, b in zip(aligned[i], aligned[j]) if a != "-" and b != "-")
            identity = matches / comparable * 100 if comparable > 0 else 0
            log += f"- {ids[i]} vs {ids[j]}: {identity:.1f}% identity ({matches}/{comparable} positions)\n"

    # Consensus sequence
    consensus_seq = "".join(consensus)
    log += f"\n## Consensus Sequence\n\n"
    for i in range(0, len(consensus_seq), 60):
        chunk = consensus_seq[i:i+60]
        log += f"  {i+1:>5}  {chunk}\n"

    # Conservation score distribution
    high = sum(1 for s in conservation_scores if s > 0.8)
    medium = sum(1 for s in conservation_scores if 0.5 < s <= 0.8)
    low = sum(1 for s in conservation_scores if 0 < s <= 0.5)
    gaps_only = sum(1 for s in conservation_scores if s == 0)

    log += f"\n## Conservation Score Distribution\n\n"
    log += f"- High (>80%): {high} positions ({high / max_len * 100:.1f}%)\n"
    log += f"- Medium (50-80%): {medium} positions ({medium / max_len * 100:.1f}%)\n"
    log += f"- Low (<50%): {low} positions ({low / max_len * 100:.1f}%)\n"
    if gaps_only > 0:
        log += f"- Gaps only: {gaps_only} positions\n"

    log += "\n## Summary\n\n"
    log += f"Conservation analysis completed for {len(ids)} sequences.\n"
    log += f"Identified {len(conserved_positions)} highly conserved positions out of {max_len} total.\n"

    return log
$$;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Smoke Tests
-- MAGIC Quick validation that each function runs without errors.

-- COMMAND ----------

-- Test 1: Circular Dichroism
SELECT biomni.agent.analyze_circular_dichroism_spectra(
    'TestProtein',
    'protein',
    ARRAY(190.0, 192.0, 195.0, 200.0, 210.0, 215.0, 218.0, 220.0),
    ARRAY(5.0, 4.0, 3.0, -2.0, -1.0, -4.0, -3.5, -3.0),
    ARRAY(),
    ARRAY()
) AS cd_result;

-- COMMAND ----------

-- Test 2: RNA Secondary Structure
SELECT biomni.agent.analyze_rna_secondary_structure_features(
    '(((...)))',
    'GGGAAACCC'
) AS rna_result;

-- COMMAND ----------

-- Test 3: Protease Kinetics
SELECT biomni.agent.analyze_protease_kinetics(
    ARRAY(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
    '[[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], [3.0, 5.0, 7.0, 9.0, 10.5, 11.5, 12.0, 12.5, 13.0, 13.5]]',
    ARRAY(5.0, 25.0, 100.0),
    0.01
) AS protease_result;

-- COMMAND ----------

-- Test 4: Enzyme Kinetics Assay
SELECT biomni.agent.analyze_enzyme_kinetics_assay(
    'Trypsin',
    ARRAY(1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0),
    50.0,
    '{"Aprotinin": [0, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}',
    ARRAY()
) AS enzyme_result;

-- COMMAND ----------

-- Test 5: ITC Binding Thermodynamics
SELECT biomni.agent.analyze_itc_binding_thermodynamics(
    ARRAY(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    ARRAY(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
    ARRAY(-10.0, -9.5, -8.0, -6.0, -4.0, -2.5, -1.5, -1.0, -0.5, -0.3),
    298.15,
    0.0001,
    0.001
) AS itc_result;

-- COMMAND ----------

-- Test 6: Protein Conservation
SELECT biomni.agent.analyze_protein_conservation(
    '[{"id": "Human", "sequence": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"}, {"id": "Mouse", "sequence": "MEESQAELGVEPPLSQETFSDLWKLLPPNNVLSTLPSSDSIEELFLSENVTGWLEDPGTT"}, {"id": "Chicken", "sequence": "MEDSQAELGVEPPLSQETFSDLWKLLPENNVLSDEVSQAMDDLMLSPDDLAQWLTEDPGP"}]'
) AS conservation_result;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Grant Permissions (customize as needed)
-- MAGIC
-- MAGIC ```sql
-- MAGIC -- Grant to a specific user or group
-- MAGIC GRANT USAGE ON CATALOG biomni TO `<user-or-group>`;
-- MAGIC GRANT USAGE ON SCHEMA biomni.agent TO `<user-or-group>`;
-- MAGIC GRANT EXECUTE ON SCHEMA biomni.agent TO `<user-or-group>`;
-- MAGIC ```
