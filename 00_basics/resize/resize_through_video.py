
sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

import cv2

import time



# --------------------------------------------------------------------------------------
# getResize()
# --------------------------------------------------------------------------------------

def getResize(src):
    """CPUを使用"""
    basePixSize = 720  # 縦横で大きい辺の変更したいサイズ
    height = src.shape[0]
    width = src.shape[1]

    largeSize = max(height, width)  # 大きい方の辺のサイズ
    resizeRate = basePixSize / largeSize  # 変更比率を計算

    dst = cv2.resize(src, (int(width * resizeRate), int(height * resizeRate)),
                     interpolation=None)

    return dst



# --------------------------------------------------------------------------------------
# VideoCapture
# --------------------------------------------------------------------------------------

cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)


cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



# ----------
if not cap.isOpened():  # ビデオキャプチャー可能か判断
    print("Not Opened Video Camera")
    exit()

while True:
    ret, img = cap.read()
    if ret == False:  # キャプチャー画像取得に失敗したら終了
        print("Video Capture Err")
        break

    # ここで処理を実行する
    img = getResize(img)

    cv2.imshow("Final result", img)  # 画面表示
    if cv2.waitKey(10) > -1:
        break



# ----------
cap.release()

cv2.destroyAllWindows()

