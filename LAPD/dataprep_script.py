import pandas as pd
from utils import do_basic_cleaning
from dataprep import eda

df = pd.read_csv('data/Crime_Data_from_2020_to_Present.csv')

df = do_basic_cleaning(df)

eda.create_report(df).save('eda_report.html')