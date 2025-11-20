import urllib.request, zipfile

url = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
urllib.request.urlretrieve(url, "E:/song_project/data/fma_metadata.zip")

with zipfile.ZipFile("E:/song_project/data/fma_metadata.zip", "r") as z:
    z.extractall("E:/song_project/data/fma_metadata")
print("[DONE] FMA metadata downloaded and extracted.")