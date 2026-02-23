import pandas as pd
import requests
import PIL.Image
from io import BytesIO # Helps open image
from IPython.display import display, clear_output # For displaying progress
import os # helps manage file paths

df = pd.read_csv("UCSC_iNat_observations.csv")

df['common_name'] = df['common_name'].str.lower() #Data cleaning

# Add species table, specific species limit tracker
speciesTable = pd.read_csv("finalSpeciesTable.csv")
desiredSpeciesList = speciesTable['scientific_name']

speciesLimit = 1000000000 # set this number so it will move on to other observations after seeing this number from a species
speciesLimTracker = pd.DataFrame(columns = ['scientific_name', 'count_processed'])

speciesLimTracker['scientific_name'] = desiredSpeciesList
speciesLimTracker['count_processed'] = 0
speciesLimTracker = speciesLimTracker.set_index('scientific_name') # This allows the code to check the counts based on searching for the row of the scientific name

'''Downloading logic'''

# [off] Only process a subset of the rows for testing (note that rows with other species not in speciesList will be skipped)
# df = df.loc[:200, :]

directory = 'images_all/'
total = 0

for i, (idx, row) in enumerate(df.iterrows()):
    
# Check that observed species is desired and we haven't already downloaded too many from that species
    if row['scientific_name'] in list(desiredSpeciesList) and speciesLimTracker.loc[row['scientific_name'], 'count_processed'].item() < speciesLimit:
        
        # Assign a variable to make it cleaner
        rowSciName = row['scientific_name']
        thisSpeciesCount = speciesLimTracker.loc[row['scientific_name'], 'count_processed'].item()
    
        # Code modified from https://github.com/hans-elliott99/toxic-plant-classification/blob/main/notebooks/scrape-iNaturalist.ipynb
        url = row['image_url']

        total_string = f"{total:05d}_{rowSciName}"
        img_name = os.path.join(total_string+".jpg")
        img_save_path = os.path.join(directory+img_name)
        
        try: #try the url, if successful open the image and save to the img_save_path
            result = requests.get(url, timeout=60)
            img = PIL.Image.open(BytesIO(result.content))
            img.save(img_save_path)
            
            # update the counter
            total += 1

            # record image name and path to our data frame for future export
            df.loc[idx, 'img_name'] = img_name
            df.loc[idx, 'img_path'] = img_save_path

            # Add 1 to the species tracker
            speciesLimTracker.loc[row['scientific_name'], 'count_processed'] = thisSpeciesCount + 1

            display(f"[INFO] downloaded: {img_save_path} | Total = {total} | Total {rowSciName} = {thisSpeciesCount}")

        except Exception as e:
            display(f"[INFO] error downloading {img_save_path}...skipping")
            display(e)
    else:
        print("entry", i, ":", row['scientific_name'], ", is not in species list or is already covered .... skipped!")

# Order columns for final data frame
column_order = ['img_name', 'img_path', 'scientific_name', 'id', 'observed_on', 'url', 'image_url', 'common_name', 
                'taxon_family_name', 'taxon_genus_name',
                ]
df = df[column_order]
df = df.rename(columns = {"id" : "iNat ID", "url" : "observation_url"})
df_cleaned = df.dropna(subset=['img_path'])

# Export with original data but also local file paths to images
df.to_csv("UCSC_iNat_observations_downloaded.csv", index = False)
df_cleaned.to_csv("UCSC_iNat_observations_downloads_only.csv")

print("done\n\n")
