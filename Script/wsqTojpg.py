import sys
from PIL import Image
import wsq


srcPath = sys.argv[1]
destPath=srcPath.lower().replace("wsq", "jpg")

img = Image.open(srcPath)

img2 = img.convert("RGB")
img2.save(destPath)
