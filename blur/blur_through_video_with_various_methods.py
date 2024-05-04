
sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

import cv2

import time



# --------------------------------------------------------------------------------------
# VideoCapture and blurring with various methods
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

    if not ret:  # キャプチャー画像取得に失敗したら終了
        print("Video Capture Err")
        break

    # ----------
    # ksize should be odd number
    # img = cv2.blur(src=img, ksize=(15,15))


    # ----------
    # img = cv2.GaussianBlur(src=img, ksize=(15,15), sigmaX=0, sigmaY=0)

    # ----------
    # img = cv2.medianBlur(src=img, ksize=15)

    # ----------
    # バイラテラルフィルタ
    # 注目画素の近くにあるものをより重視して反映させる処理
    # src: 入力画像
    # d: 注目画素をぼかすために使われる領域  0の場合sigmaSpace から求められる
    # sigmaColor: 色の標準偏差。
    # (この値を大きくすると全体的に均一に混ぜ合わされ結果として同じような色の領域がより大きくなり,大きくし過ぎるとガウシアンフィルタと違いがなくなる)
    # sigmaSpace: 距離の標準偏差。(このパラメータが大きくなると，より遠くのピクセル同士が影響を及ぼす)

    img = cv2.bilateralFilter(src=img, d=15, sigmaColor=40, sigmaSpace=40)

    cv2.imshow("Final result", img)  # 画面表示
    if cv2.waitKey(10) > -1:
        break



# ----------
cap.release()

cv2.destroyAllWindows()
