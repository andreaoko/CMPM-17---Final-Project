import pandas as pd

df = pd.read_csv('getImages2/UCSC_iNat_observations_downloads_only.csv')

counts = df['taxon_family_name'].value_counts()

counts.to_csv("getImages2/family_counts.csv", index = True)

print(counts.info())