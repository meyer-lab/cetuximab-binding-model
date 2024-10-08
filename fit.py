from pathlib import Path

import numpy as np
import pandas as pd

from binding_model import infer_Rbound_batched

THIS_DIR = Path(__file__).parent

RCPS = [
    # "FcgRIIA-131R",
    # "FcgRIIB",
    "FcgRIIIA-158V",
    # "FcgRIIIB",
]
AFFINITY_COLS = [f"log_aff_{r}" for r in RCPS]
ABUNDANCE_COLS = [f"log_abund_{r}" for r in RCPS]


def load_data():
    """Load the entire dataset and perform some preprocessing."""
    data = pd.read_excel(THIS_DIR / "20240515_MSDcytokines_tensor_dataset.xlsx")
    return data


def init_optimize_df(experiment_df: pd.DataFrame) -> pd.DataFrame:
    opt_df = experiment_df.copy()

    # drop rows with no NK cells
    opt_df = opt_df[opt_df["Cells"] == "NK cells + A549"]
    opt_df.drop(columns=["Cells"], inplace=True)

    opt_df.rename(
        columns={
            "Incubation time (hr)": "time",
            "Cetuximab Variant": "variant",
            "Cetuximab Concentration (ug/ml)": "conc",
            "Response": "signal",
            "Donor": "donor",
            "Cytokine": "cytokine",
        },
        inplace=True,
    )

    # remove NaN signals
    opt_df = opt_df[~opt_df["signal"].isna()]

    baselines = (
        opt_df[opt_df["variant"] == "-"].groupby(["donor"])["signal"].mean()
    ).rename("baseline_signal")
    # subtract the baseline signal from the signal, making sure the donor and time are aligned
    # merge the baseline signal into the opt_df
    opt_df = opt_df.merge(baselines, on=["donor"], how="left", validate="many_to_one")
    # subtract the baseline signal from the signal
    opt_df["signal"] = opt_df["signal"] - opt_df["baseline_signal"]
    # drop the baseline signal column
    opt_df.drop(columns=["baseline_signal"], inplace=True)

    # drop the baseline rows
    opt_df = opt_df[opt_df["variant"] != "-"]

    # if signal is negative, set to 0
    opt_df["signal"] = opt_df["signal"].clip(lower=0)

    opt_df["conc"] = opt_df["conc"].astype(np.float64) * 1e-9

    # import affinities for each variant
    affinities = pd.read_excel(THIS_DIR / "affinity.xlsx")
    affinities = affinities[["variant", *RCPS]]
    # prepend "log_aff_" to the columns (except for variant)
    affinities.rename(columns={col: f"log_aff_{col}" for col in RCPS}, inplace=True)
    # log the affinities
    affinities[AFFINITY_COLS] = affinities[AFFINITY_COLS].apply(np.log10)
    opt_df = opt_df.merge(affinities, on="variant")

    opt_df["eff_cancer_cell_valency"] = 5.0

    opt_df["log_rbound_signal_coeff"] = -2.0

    # initialize the receptor abundances
    # for receptor in RCPS:
    #     opt_df[f"log_abund_{receptor}"] = 5.0
    opt_df["log_abund_FcgRIIIA-158V"] = np.log10(2e3)

    opt_df["log_KxStar"] = -12.0

    return opt_df


def infer_signal(opt_df: pd.DataFrame) -> np.ndarray:
    """
    Infers the signal based on the optimized parameters.

    Args:
        opt_df: DataFrame containing optimized parameters.

    Returns:
        Array of inferred signals.
    """
    Rtot = 10 ** opt_df[ABUNDANCE_COLS].values
    valency = (opt_df["eff_cancer_cell_valency"]).values[:, None, None]
    Ka = 10 ** opt_df[AFFINITY_COLS].values[:, None, :]
    Rbound = infer_Rbound_batched(
        opt_df["conc"].values,
        10 ** opt_df["log_KxStar"],
        Rtot,
        valency,
        np.ones((opt_df.shape[0], 1)),
        Ka,
    )
    return Rbound[:, 0] * 10 ** opt_df["log_rbound_signal_coeff"].values


const_params = [
    "conc",
    "log_KxStar",
    "signal",
    *AFFINITY_COLS,
    "donor",
    "time",
    "variant",
    "cytokine",
    "Tube #",
    "log_abund_FcgRIIIA-158V",
]

var_params = [
    "log_rbound_signal_coeff",
    "eff_cancer_cell_valency",
]

# each var_param can be stratified by a constant_param or entire dataset
stratifiers = {
    "log_rbound_signal_coeff": "cytokine",
}

param_bounds = {
    "log_rbound_signal_coeff": (-9.0, 5),
    "eff_cancer_cell_valency": (0, 100),
}
