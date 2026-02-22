import pandas as pd
import requests
import PIL.Image
from io import BytesIO # Helps open image
from IPython.display import display, clear_output # For displaying progress
import os # helps manage file paths

df = pd.read_csv("UCSC_iNat_observations.csv")

df['common_name'] = df['common_name'].str.lower()

counts = df['common_name'].value_counts()

counts.to_csv("species_counts.csv", index = True)

print(counts.info())

#Subset of file <- Important! Whole CSV is 10k images!
df = df.loc[:20, :]

directory = 'images_all/'
total = 0


# Code modified from https://github.com/hans-elliott99/toxic-plant-classification/blob/main/notebooks/scrape-iNaturalist.ipynb
for i, (idx, row) in enumerate(df.iterrows()):
    url = row['image_url']
    common_name = row['common_name']

    total_string = f"{total:05d}_{common_name}"
    img_name = os.path.join(total_string+".jpg")
    img_save_path = os.path.join(directory+img_name)
    
    try: #try the url, if successful open the image and save to the img_save_path
        result = requests.get(url, timeout=60)
        img = PIL.Image.open(BytesIO(result.content))
        img.save(img_save_path)
        # update the counter
        total += 1

        df.loc[idx, 'img_name'] = img_name
        df.loc[idx, 'img_path'] = img_save_path

        clear_output(wait=True) ##allow print statements to overwrite previous ones
        display(f"[INFO] downloaded: {img_save_path} | Total {total}")

    except Exception as e:
        display(f"[INFO] error downloading {img_save_path}...skipping")
        display(e)
    

print(df.info())

column_order = ['img_name', 'img_path', 'common_name', 'id', 'observed_on', 'url', 'image_url', 'scientific_name', 
                'taxon_family_name', 'taxon_genus_name',
                ]

df = df[column_order]

df = df.rename(columns = {"id" : "iNat ID", "url" : "observation_url"})

df.to_csv("UCSC_iNat_observations_downloaded.csv", index = False)

print("done\n\n")
