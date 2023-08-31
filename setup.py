import gdown
import zipfile

# for downloading large files with gdown, link must have &confirm=t at the end
url = "https://drive.google.com/u/0/uc?id=11ozVs6zByFjs9viD3VIIP6qKFgjZwv9E&confirm=t"
output = "screws.zip"
ret = gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as dir:
    dir.extractall()
print("Successfully pulled dataset")