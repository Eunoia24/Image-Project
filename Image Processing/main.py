"""
※ 프로그램에 대한 간략한 설명
main.py에서 종료(q,Q입력)할 때까지 키보드 및 마우스 입력을 받습니다.
main.py에서 회선만을 적용할 원본 이미지와 화면 출력용(텍스트, 포인터) 오버레이 이미지를 만들어줍니다.
회선은 Conv_Algorithm.py를 사용하여 원본 이미지에 적용시켜줍니다. (회선 진행시 넘겨줄 인자 = Image, Mask, Conv_type)
q(Q) 종료 버튼이 입력되면 회선이 진행된 이미지와, 회선된 부분을 검정색 처리한 이미지 2개를 저장하고 마무리합니다.
"""


import numpy as np, cv2
import ctypes, Conv_Algorithm
# ↑ 마우스 포인터 제거용 모듈과, 가독성을 위해 따로 분리한 회선 알고리즘 모듈


img = cv2.imread("input.jpg", cv2.IMREAD_COLOR)                                                 # 영상 불러오기
if img is None: raise Exception("영상파일 읽기 오류")                                             # 영상 읽기 오류 발생시

""" Set Variable """
ix, iy = None, None                                                                             # 드래그 용도
conv_type, conv_on = None, False                                                                # 선택한 회선, 드래그 여부
mask = np.zeros((1080, 1920, 1), np.uint8)                                                      # 마스크 생성 (회선이 적용된 픽셀 위치)
overlay, key, tmp_key = img.copy(), -1, -1                                                      # 화면 표시용 이미지, 키보드 입력값, 드래그 중 키 입력 방지용 변수
conv_list = ['1.Blurring', '2.Sharpening', '3.Prewitt', '4.Sobel', '5.Laplacian(8dir)']         # 회선 목록

""" Save Photo ( Exit Program ) """
def save_photo():
    global img, mask
    cv2.imwrite("20212041_1.jpg", img, (cv2.IMWRITE_JPEG_QUALITY, 100))                         # 회선 적용된 이미지 저장
    tmp = np.ones((1080, 1920, 1), np.uint8)                                                    # mask (0,1) 반전용 데이터 생성
    tmp = cv2.bitwise_xor(mask, tmp) * 255                                                      # mask (0,1) 반전, 1 => 255로 변환
    tmp = np.repeat(np.array(tmp, dtype=np.uint8)[:, :, np.newaxis], 3, -1)                     # tmp channel : 1 -> 3
    cv2.imwrite("20212041_2.jpg", cv2.bitwise_and(img, tmp), (cv2.IMWRITE_JPEG_QUALITY, 100))   # (img) bitwise_and (mask), 회선이 적용된 픽셀을 지운 이미지 저장
    print("저장 완료!!")

""" Overlay Convolution Text """
def change_convtext(key):
    global overlay
    default_color, select_color = (240, 240, 240), (30, 10, 100)                                # 텍스트 색상
    seq, pos = (49, 50, 51, 52, 53), ((13, 40), (145, 40), (317, 40), (438, 40), (540, 40))     # 아스키 코드(1~5), 텍스트 위치
    for i in range(5):                                                                          # 회선 텍스트 표시
        cv2.putText(overlay, conv_list[i], pos[i], cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, select_color if key == seq[i] else default_color, 3)

""" Execute Convolution (Use Conv_Algorithm Module) """
def conv(image, mask, conv_type):
    expression = 'Conv_Algorithm.' + conv_type + '(image, mask)'
    return eval(expression)                                                                     # Conv_Algorithm.py 모듈을 사용하여 회선을 실행

""" Mouse Callback Event """
def onMouse(event, x, y, flags, param):
    global img, mask, overlay, conv_on, conv_type, ix, iy
    ctypes.windll.user32.ShowCursor(False)                                                      # 기본 마우스 커서(십자 모양) 제거

    if event == cv2.EVENT_LBUTTONDOWN:                                                          # LBUTTONDOWN -> Drag On
        ix, iy = x, y
        conv_on = True

    elif event == cv2.EVENT_LBUTTONUP:                                                          # LBUTTONUP   -> Drag Off + ↓
        conv_on = False
        if conv_type:                                                                           # Execute Convolution (On Original Image)
            img = conv(img, mask, conv_type)
            overlay = img.copy()

    # (Drag) or (Mouse Move)
    # (Drag == Click Moving)
    # (Mouse Move == No Click Moving)
    elif event == cv2.EVENT_MOUSEMOVE:
        # Drag (Choose conv_type O) -> Set Scope
        if conv_on and conv_type:
            # 회선 범위 지정 (circle area)
            # cv2.circle 사용 시 빠른 드래그를 할 때 지나온 부분이 누락되는 문제 발생
            # 해결 방법 1 : np.linspace(선형보간) 사용 // 결과 : 연산량이 매우 많아 렉 발생
            # 해결 방법 2 : cv2.line 사용 // 결과 : 문제 해결
            cv2.line(mask, (ix, iy), (x, y), 1, 60)                                             # 이전 Frame에 받은 pixel pos = (ix, iy)에서 현재 pixel pos = (x, y)로 cv.line
            cv2.line(overlay, (ix, iy), (x, y), (0, 0, 150), 60)                                # 위와 동일. (위 줄은 Mask에 입력할 용도, 현재 줄은 화면 출력 용도.)
            cv2.imshow("IMAGE", cv2.addWeighted(overlay, 0.6, img, 0.4, 0))
            ix, iy = x, y

        # 1. Mouse Move -> Overlay Pointer
        # 2. Drag (Choose conv_type X) -> Overlay Pointer + Print Alert Message
        else:
            if conv_on and (not conv_type):                                                     # conv_type을 지정하지 않고 드래그 시에는 콘솔창에 경고문을 출력
                print("적용할 회선을 고르시오.")
            cv2.circle(overlay, (x, y), 30, (0, 0, 150), -1)                                    # Print Screen (Overlay Text + Pointer)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, overlay)
            change_convtext(key)
            cv2.imshow("IMAGE", overlay)


""" Main (Mouse, Keyboard) """
change_convtext(0); cv2.imshow("IMAGE", overlay)                                                # Initial Screen
cv2.setMouseCallback('IMAGE', onMouse)                                                          # Mouse Event
while True:
    key = cv2.waitKey(0)                                                                        # Keyboard Event

    if not(conv_on):                                                                            # 드래그 중 키 변경 방지
        tmp_key = key                                                                           # Drag를 시작하면 tmp_key에 기존 key값을 저장
    key = tmp_key                                                                               # Drag가 끝나면 Drag 중 key가 변경되었어도 기존 key로 복원

    if not(conv_on) and key == (ord('q') or ord('Q')):                                          # (q or Q) 입력시 종료 (Drag X인 경우에만)
        save_photo()                                                                            # Save Photo(1,2)
        break                                                                                   # Program Exit

    elif not(conv_on) and key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:            # (1~5) 입력시 회선 변경 (Drag X인 경우에만)
        change_convtext(key)                                                                    # Selected Conv => (Red Text), Rest Conv => (Black Text)
        conv_type = (conv_list[key-49][2:] if key != ord('5') else conv_list[key-49][2:-6])     # 회선(str) 변수 (onMouse -> conv func)
        cv2.imshow("IMAGE", overlay)                                                            # Print Screen (Text Color Changed)