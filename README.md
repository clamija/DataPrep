## DataPrep

Aplikacija za automatsku pripremu tabelarnih podataka (cleaning i preprocessing),
uz logističku regresiju kao baseline model. Statistike za imputaciju i IQR
računaju se samo na trening skupu, zatim se primjenjuju na test.

## Pokretanje

```bash
cd project
pip install -r requirements.txt
streamlit run app.py
```

Aplikaciju pokreni iz foldera `project`, jer se skupovi učitavaju s putanja
`data/titanic.csv` i `data/telco.csv`.

## Šta aplikacija radi

- učitavanje Titanic ili Telco skupa, ili upload vlastitog CSV-a (custom),
- prikaz broja redova i kolona,
- cleaning i preprocessing,
- logistička regresija,
- poređenje BASELINE vs CLEANED (isti test redovi),
- Data Quality Score: RAW vs CLEANED (iste kolone, prije i poslije čišćenja).

## Prilagodba na novi skup

U `config.py` postavi:

- `target_column`
- `drop_columns`
- opcionalno `value_mappings`

Glavna logika pipeline-a ostaje ista.

## Profili u konfiguraciji

- Telco: `target_column = "Churn"`, `drop_columns = ["customerID"]`
- Titanic: `target_column = "Survived"`, `drop_columns = ["PassengerId", "Name", "Ticket"]`
