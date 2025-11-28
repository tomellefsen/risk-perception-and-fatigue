"""
Cleans raw INSPQ COVID-19 case data [1].

This script performs the following steps:
1.  Loads the raw case data from ``data/2_cas_confirme_quo.csv``.
2.  Filters the data to keep only the 'Total' aggregate rows.
3.  Removes rows with an 'Date inconnue' (Unknown date).
4.  Removes rows related to 'sexe' (sex) groupings.
5.  Drops superfluous columns ('Regroupement', 'Croisement', 'Nom').
6.  Renames 'Date' to 'date' and 'psi_quo_pos_n' to 'cases'.
7.  Saves the cleaned, simplified data to ``data/cases.csv``.

Notes
-----
This script is intended to be run directly to process the raw data file
downloaded from the INSPQ Before any analysis.

Data Source:
    Institut national de santé public du Québec
    https://www.inspq.qc.ca/covid-19/donnees/archives
    Accessed: November 6 2025
"""

import pandas as pd

df = pd.read_csv("data/2_cas_confirme_quo.csv")

df = df[df["Nom"] == "Total"]
df = df[df["Date"] != "Date inconnue"]
df = df[~df["Regroupement"].str.contains("sexe", case=False, na=False)]
df = df.drop(columns=["Regroupement", "Croisement", "Nom"], errors="ignore")
df = df.rename(columns={"Date": "date", "psi_quo_pos_n": "cases"})

df.to_csv("data/cases.csv", index=False)