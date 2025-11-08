# Data downloaded from the Institut national de santé public du Québec
# Accessed: November 6 2025
# https://www.inspq.qc.ca/covid-19/donnees/archives

import pandas as pd

df = pd.read_csv("data/2_cas_confirme_quo.csv")

df = df[df["Nom"] == "Total"]
df = df[df["Date"] != "Date inconnue"]
df = df[~df["Regroupement"].str.contains("sexe", case=False, na=False)]
df = df.drop(columns=["Regroupement", "Croisement", "Nom"], errors="ignore")
df = df.rename(columns={"Date": "date", "psi_quo_pos_n": "cases"})

df.to_csv("data/cases.csv", index=False)

# 