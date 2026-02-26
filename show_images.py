import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import PIL.Image
from io import BytesIO                                      #Helps open image
import requests
from torchvision.transforms import v2

df = pd.read_csv("data/UCSC_iNat_observations_downloads_only.csv")                                                  #place the downloaded images into a pandas dataframe


num_images = 100


for i in range(num_images):
    url = df.iloc[i]['image_url']                                                                                   #locate image url, common name, and scientific name
    name = df.iloc[i]['common_name']
    sci_name = df.iloc[i]['scientific_name']

    result = requests.get(url, timeout=60)                                                                          #requests the image
    img = PIL.Image.open(BytesIO(result.content))                                                                   #opens the image

    size = v2.Resize((500,500))                                                                                     #resize the image
    img_resize = size(img)

    plt.subplot(10, 10, i+1)                                                                                        #plot the image in a 10 x 10 grid
    plt.imshow(img_resize)                                                                                          #sets all images to the same size
    plt.axis("off")

plt.tight_layout()                                                                                                  # this line of code makes the layout/format nice with even spacing
plt.show()