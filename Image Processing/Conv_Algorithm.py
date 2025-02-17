## 회선 방법
## 드래그된 영역에 해당하면서 동시에 회선이 진행되지 않은 영역에 대해서만 pixel 위치값(pos)을 구해줍니다.
## 이후 각 픽셀의 roi(3x3)를 구하여 cv2.multiply와 cv2.sumElems, cv2.magnitude 등의 함수를 사용해 결과값을 구하고 변경해줍니다.

## 테두리 픽셀의 회선 진행 시에 필터의 범위를 벗어나는 문제 해결 방법 => ( cv2.copyMakeBorder(cv2.BORDER_REPLICATE) 함수 사용 )
## 테두리 픽셀값을 그대로 복사한 패딩을 1칸 추가하여 해결 (필터 크기를 3x3으로 잡았으므로 패딩도 1로 설정)

## 특정 회선은 필터의 종류가 다양하거나 교재에 틀린 내용이 있어 따로 사용한 필터값을 주석으로 정리
"""
Shapening = [[-1, -1, -1],
             [-1, +9, -1],
             [-1, -1, -1]]

Prewitt = Gx: [[-1, +0, +1],      Gy: [[-1, -1, -1],
               [-1, +0, +1],           [+0, +0, +0],
               [-1, +0, +1]]           [+1, +1, +1]]

Sobel =   Gx: [[-1, +0, +1],      Gy: [[-1, -2, -1],
               [-2, +0, +2],           [+0, +0, +0],
               [-1, +0, +1]]           [+1, +2, +1]]
"""


import numpy as np, cv2


""" Conv Pixel Check """
# Bitwise operation between (Already Convoluted pixels) and (The Dragged pixels)
old_data = np.zeros((1080, 1920, 1), np.uint8)                                              # Already Convoluted pixels
def Mask(new_data):
    global old_data
    tmp = cv2.bitwise_xor(old_data, new_data).reshape((1080, 1920, 1))                      # 회선을 진행할 픽셀 영역
    old_data = cv2.bitwise_or(old_data, new_data).reshape((1080, 1920, 1))                  # old_data = (old) bitwise_or (new)
    return tmp

""" Convolution """
# Blurring, Sharpening
def filtering(image, pos, mask):
    # 1. Convolution 된 값을 반영할 복사본
    dst = image.copy().astype('float32')
    # 2. Convolution
    for i, j in zip(pos[0], pos[1]):
        roi = image[i:i + 3, j:j + 3, :].astype('float32')                                  # padding을 추가했으므로 i(j):i+3(j+3)
        tmp = cv2.multiply(roi, mask)
        dst[i+1, j+1, :] = cv2.sumElems(tmp)[:3]
    # Remove padding + Abs + uint8
    return cv2.convertScaleAbs(dst[1:-1, 1:-1, :])

# Prewitt, Sobel, Laplacian
def differential(image, pos, mask1, mask2):
    # 1. Convolution 된 값을 반영할 복사본
    dst = image.copy().astype('float32')
    # 2. Convolution
    for i, j in zip(pos[0], pos[1]):
        roi = image[i:i + 3, j:j + 3, :].astype('float32')                                  # padding을 추가했으므로 i(j):i+3(j+3)
        tmp1 = cv2.multiply(roi, mask1)
        tmp2 = cv2.multiply(roi, mask2)
        dst[i+1, j+1, :] = cv2.magnitude(cv2.sumElems(tmp1)[:3], cv2.sumElems(tmp2)[:3])[:, 0]
    # Remove padding + Abs + uint8
    return cv2.convertScaleAbs(dst[1:-1, 1:-1, :])

""" Convolution Type """
def Blurring(image, mask):
    # 1. conv_mask
    conv_mask = np.full((3, 3, 3), 1/9, np.float32) # (3, 3, 3)
    # 2. padding
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    # 3. get pixel position (drag area, use mask)
    mask = Mask(mask)
    rows, cols = np.where(mask[:, :, 0] > 0)
    # 4. Blurring
    return filtering(image, (rows, cols), conv_mask)

def Sharpening(image, mask):
    # 1. conv_mask
    data = [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
    conv_mask = np.repeat(np.array(data, dtype=np.float32)[:, :, np.newaxis], 3, -1) # (3, 3, 3)
    # 2. padding
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    # 3. get pixel position (drag area, use mask)
    mask = Mask(mask)
    rows, cols = np.where(mask[:, :, 0] > 0)
    # 4. Sharpening
    return filtering(image, (rows, cols), conv_mask)

def Prewitt(image, mask):
    # 1. conv_mask
    conv_mask_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    conv_mask_y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    conv_mask_x = np.repeat(np.array(conv_mask_x, dtype=np.float32)[:, :, np.newaxis], 3, -1)  # (3, 3, 3)
    conv_mask_y = np.repeat(np.array(conv_mask_y, dtype=np.float32)[:, :, np.newaxis], 3, -1)  # (3, 3, 3)
    # 2. padding
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    # 3. get pixel position (drag area, use mask)
    mask = Mask(mask)
    rows, cols = np.where(mask[:, :, 0] > 0)
    # 4. Prewitt
    return differential(image, (rows, cols), conv_mask_x, conv_mask_y)

def Sobel(image, mask):
    # 1. conv_mask
    conv_mask_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    conv_mask_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    conv_mask_x = np.repeat(np.array(conv_mask_x, dtype=np.float32)[:, :, np.newaxis], 3, -1)  # (3, 3, 3)
    conv_mask_y = np.repeat(np.array(conv_mask_y, dtype=np.float32)[:, :, np.newaxis], 3, -1)  # (3, 3, 3)
    # 2. padding
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    # 3. get pixel position (drag area, use mask)
    mask = Mask(mask)
    rows, cols = np.where(mask[:, :, 0] > 0)
    # 4. Sobel
    return differential(image, (rows, cols), conv_mask_x, conv_mask_y)

def Laplacian(image, mask):
    # 1. conv_mask
    conv_mask_x = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    conv_mask_y = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
    conv_mask_x = np.repeat(np.array(conv_mask_x, dtype=np.float32)[:, :, np.newaxis], 3, -1)  # (3, 3, 3)
    conv_mask_y = np.repeat(np.array(conv_mask_y, dtype=np.float32)[:, :, np.newaxis], 3, -1)  # (3, 3, 3)
    # 2. padding
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    # 3. get pixel position (drag area, use mask)
    mask = Mask(mask)
    rows, cols = np.where(mask[:, :, 0] > 0)
    # 4. Laplacian
    return differential(image, (rows, cols), conv_mask_x, conv_mask_y)