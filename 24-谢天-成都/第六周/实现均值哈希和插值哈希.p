import cv2

def aHash(img):
 
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    s = 0
    hash_str = ''
 
    for i in range(8):
        for j in range(8):
            s += img_gray[i, j]

    avg = s / 64
  
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
 
    return hash_str



def dHash(img):

    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hash_str = ''

    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > img_gray[i, j + 1]:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str



def cmpHash(hash_1, hash_2):

    n = 0

    if len(hash_1) != len(hash_2):
        return -1
 
    for i in range(len(hash_1)):
        if hash_1[i] != hash_2[i]:
            n += 1
    
    return n



img1 = cv2.imread("lenna.png")
img2 = cv2.imread("lenna_noise.png")

print(f"图像1的均值哈希值:{aHash(img1)}\n图像1的均值哈希值:{aHash(img2)}")
print(f"两个哈希值的相似度{cmpHash(aHash(img1), aHash(img2))}")
print(f"图像1的均值哈希值:{dHash(img1)}\n图像1的均值哈希值:{dHash(img2)}")
print(f"两个哈希值的相似度{cmpHash(dHash(img1), dHash(img2))}")
