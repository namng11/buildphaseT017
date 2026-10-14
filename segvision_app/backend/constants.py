"""Constants for BDD100K 19-class Semantic Segmentation."""
import numpy as np

# 19 nhãn chuẩn theo quy chuẩn của BDD100K / Cityscapes
CLASS_NAMES = [
    "road",          # 0
    "sidewalk",      # 1
    "building",      # 2
    "wall",          # 3
    "fence",         # 4
    "pole",          # 5
    "traffic light", # 6
    "traffic sign",  # 7
    "vegetation",    # 8
    "terrain",       # 9
    "sky",           # 10
    "person",        # 11
    "rider",         # 12
    "car",           # 13
    "truck",         # 14
    "bus",           # 15
    "train",         # 16
    "motorcycle",    # 17
    "bicycle"        # 18
]

# Giá trị pixel bỏ qua (không tính vào IoU)
IGNORE_INDEX = 255

ID2LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Bảng mã màu RGB chuẩn quốc tế của Cityscapes / BDD100k
COLOR_PALETTE = np.array([
    [128,  64, 128],  # 0: road (Tím xám)
    [244,  35, 232],  # 1: sidewalk (Hồng tím)
    [ 70,  70,  70],  # 2: building (Xám đậm)
    [102, 102, 156],  # 3: wall (Xám xanh)
    [190, 153, 153],  # 4: fence (Hồng đất)
    [153, 153, 153],  # 5: pole (Xám vừa)
    [250, 170,  30],  # 6: traffic light (Cam vàng)
    [220, 220,   0],  # 7: traffic sign (Vàng chanh)
    [107, 142,  35],  # 8: vegetation (Xanh rêu)
    [152, 251, 152],  # 9: terrain (Xanh nõn chuối)
    [ 70, 130, 180],  # 10: sky (Xanh da trời)
    [220,  20,  60],  # 11: person (Đỏ tươi)
    [255,   0,   0],  # 12: rider (Đỏ cờ)
    [  0,   0, 142],  # 13: car (Xanh dương đậm)
    [  0,   0,  70],  # 14: truck (Xanh hải quân)
    [  0,  60, 100],  # 15: bus (Xanh cổ vịt)
    [  0,  80, 100],  # 16: train (Xanh lam sẫm)
    [  0,   0, 230],  # 17: motorcycle (Xanh dương sáng)
    [119,  11,  32],  # 18: bicycle (Đỏ mận)
], dtype=np.uint8)

# Chuyển đổi mã RGB sang chuỗi HEX để dùng trên giao diện Web (CSS)
HEX_PALETTE = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in COLOR_PALETTE]
