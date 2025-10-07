import os
import sys
import cv2
import numpy as np
from app.core.insightface_embedder import InsightFaceEmbedder

IMG='face_data/dvd_1759471448648.jpg'
if not os.path.exists(IMG):
    print('Image missing:', IMG)
    sys.exit(1)

embedder = InsightFaceEmbedder(model_name='buffalo_l')
print('Embedder created')
img = cv2.imread(IMG)
print('Loaded image shape:', None if img is None else img.shape)
if img is None:
    sys.exit(1)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Try original
print('\n--- Original image ---')
results = embedder.detect_and_encode_faces(img_rgb)
print('detect_and_encode_faces results:', results)
emb = embedder.get_embedding(img_rgb)
print('get_embedding returned:', None if emb is None else ('embedding shape', np.asarray(emb).shape))

# Try scaled versions
for scale in [1.5, 2.0, 3.0]:
    h,w = img_rgb.shape[:2]
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    print(f'\n--- Scale {scale} -> size {nw}x{nh} ---')
    r = embedder.detect_and_encode_faces(resized)
    print('detect_and_encode_faces results:', r)
    e = embedder.get_embedding(resized)
    print('get_embedding returned:', None if e is None else ('embedding shape', np.asarray(e).shape))

# Also try cropping center and scaling
h,w = img_rgb.shape[:2]
cx, cy = w//2, h//2
s = min(w,h)//2
crop = img_rgb[cy-s:cy+s, cx-s:cx+s]
if crop.size>0:
    for scale in [1,2]:
        nh, nw = crop.shape[0]*scale, crop.shape[1]*scale
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        print(f'\n--- Center crop scaled {scale} -> {nw}x{nh} ---')
        r = embedder.detect_and_encode_faces(resized)
        print('detect_and_encode_faces results:', r)
        e = embedder.get_embedding(resized)
        print('get_embedding returned:', None if e is None else ('embedding shape', np.asarray(e).shape))

print('\nDone')
