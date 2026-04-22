import pandas as pd

import cv2
import numpy as np

def decode_image(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # HWC BGR
    return img

df = pd.read_parquet("outputs/success_0.parquet")

for i in range(len(df)):
    row = df.iloc[i]
    
    image = decode_image(row["images.cam_side"])

    cv2.imshow("AA", image)
    cv2.waitKey(1)