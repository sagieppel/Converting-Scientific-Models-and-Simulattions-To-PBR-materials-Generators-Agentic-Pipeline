from pathlib import Path
import cv2

root = r"/media/fogbrain/6TB/python_project/PBR_Materials_Generator/Selected_PBR"

for png in Path(root).rglob("*.png"):
    img = cv2.imread(str(png))
    jpg = png.with_suffix(".jpg")

    cv2.imwrite(str(jpg), img)
    png.unlink()

    print(f"Converted: {png}")

print("Done")