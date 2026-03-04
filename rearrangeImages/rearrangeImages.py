'''
Script to rearrange images into the file structure requested, ignoring families with a low count

We should each run this once, to make the corresponding image folder. Or, one of us could run it and then send the folder over google drive. 
We should not push the resulting image folder to github, as it will bloat the repository history.
'''

import pandas as pd
import PIL.Image # to open and resave image
import os # for file management

######### PART 1 ##############

existingDownloads = pd.read_csv('getImages2/UCSC_iNat_observations_downloads_only.csv')
familyList = pd.read_csv('getImages2/family_counts.csv')
familyList = familyList.set_index('taxon_family_name')
speciesList = pd.read_csv('getImages2/finalSpeciesTable.csv')
speciesList = speciesList.set_index('scientific_name')

familyMinimum = 100

directory = 'imagesOrganized/'
if not os.path.exists(directory): # Make directory if it doesn't already exist
            os.mkdir(directory)
            print(f"Folder '{directory}' created.")


# Keep track of which folders/files already exist
imagesAlreadyRearranged = 0

# Organize images by species
for i, (idx, row) in enumerate(existingDownloads.iterrows()):
    
    family = row['taxon_family_name']
    familyTotal = familyList.loc[family, 'count'].item()
    species = row['scientific_name']
    speciesTotal = speciesList.loc[species, 'count'].item()

    # Check that observed species part of a family with sufficient observations, and that species is greater than 10
    # Arctostaphylos crustacea crustacea is skipped. There are only 2 images available, despite a higher value in the table.
    if familyTotal > familyMinimum and speciesTotal >= 10 and species != 'Arctostaphylos crustacea crustacea':
        
        sciName = row['scientific_name'] # Name/organize images based on families
        folderPath = directory + family
        imagePathOriginal = 'getImages2/' + row['img_path'] # The original path didn't include the getImages2 folder
        imageName = row['img_name']

        # Check if folder exists and if not, make a new one
        # 3 lines Adapted from https://medium.com/@shahsanap89/different-ways-to-create-a-folder-in-python-38857d776d65
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
            print(f"Folder '{folderPath}' created.")
        
        # Check if the specific image name exists in that folder
        imagePathNew = folderPath + '/' + imageName
        existingDownloads.loc[idx, 'img_path_new'] = imagePathNew # save new image path 

        if not os.path.exists(imagePathNew):
            
            # Save the new image
            img = PIL.Image.open(imagePathOriginal)
            img.save(imagePathNew)

        else:
            imagesAlreadyRearranged += 1
    
    if i > 10000: # Just in case the loop gets stuck somehow
        exit()

existingDownloads = existingDownloads.dropna()
existingDownloads.to_csv('DownloadedImageData_NewPaths.csv', index = False)

print(f'{imagesAlreadyRearranged} images already saved in new locations')
print('New file paths appended to dataframe, see DownloadedImageData_NewPaths.csv')

############## PART 2 - Split Images ######################

import splitfolders

inputFolder = 'imagesOrganized'
outputFolder = 'imagesOrganizedSplit'
if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
            print(f"Folder '{outputFolder}' created.")

splitfolders.ratio(inputFolder, outputFolder, 
						ratio=(.8, .1, .1))