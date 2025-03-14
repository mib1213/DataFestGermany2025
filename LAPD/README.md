### How to run LAPD notebooks?

- Download the files `Crime_Data_2020_to_Present.csv` and `mocodes.csv` from [THI Sharepoint](https://thide-my.sharepoint.com/:f:/r/personal/mib1213_thi_de/Documents/DataFestGermany2025?csf=1&web=1&e=lsVGZK).
- Save these files in the `data` folder inside the `DataFestGermany2025`.
- Once you run the `1_data_clearning.ipynb`, you will get the cleaned data in 2 different formats as a .pkl file, one is `crime_data.pkl` and the other is `exploded_crime_data.pkl`. This .pkl file can easily be imported in Pandas for data analysis as it will keep the records of datatypes.
- You can also look into the already generated EDA report using dataprep called `eda_report.html` in OneDrive [here](https://thide-my.sharepoint.com/:f:/r/personal/mib1213_thi_de/Documents/DataFestGermany2025/LAPD?csf=1&web=1&e=FpucEq). If you want to generate your own dataprep EDA report, run the script `dataprep_script.py`.
- `utils.py` contains some helper functions.