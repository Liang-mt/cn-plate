import cv2
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import hyperlpr3 as lpr3

def cv2ImgAddText(img, font_path, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(font_path, textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
def draw_plate_on_image(img, dict_list, font_path):
    if not dict_list:
        return img

    for result_dict in dict_list:

        x1, y1, x2, y2 = result_dict["box"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (139, 139, 102), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 - 20), (x2, y1), (139, 139, 102), -1)

        left = x1 + 1
        top = y1 - 18

        img = cv2ImgAddText(img, font_path, result_dict["text"], left, top, (255, 255, 255), 20)

    return img

def get_plate(text,code, confidence, type_idx, box):
    result_dict = {}
    result_dict['code'] = code  # 车牌
    result_dict['confidence'] = confidence  # 置信度
    result_dict['text'] = text
    result_dict['type_idx'] = type_idx  # 人脸角点坐标
    result_dict['box'] = box  # 识别框坐标
    return result_dict

def recognize_plate(frame, catcher):
    dict_list = []

    # 执行识别算法
    results = catcher(frame)
    for code, confidence, type_idx, box in results:

        text = f"{code} - {confidence:.2f}"
        result_dict = get_plate(text,code, confidence, type_idx, box)
        dict_list.append(result_dict)


    return dict_list


if __name__ == '__main__':
    input_video = "./video/1.mp4"
    output_video = "./video/output.mp4"  # 使用 .mp4 扩展名
    font_path = "./platech.ttf" #字体路径

    # 读取视频
    cap = cv2.VideoCapture(input_video)  # 替换成你的视频文件路径

    catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_LOW)

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出视频编码格式
    fps = int(cap.get(5))
    width = int(cap.get(3))
    height = int(cap.get(4))
    output_video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))  # 替换成你的输出视频路径和参数

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # 对每一帧进行识别和绘制
        dict_list = recognize_plate(frame, catcher)
        frame = draw_plate_on_image(frame, dict_list, font_path)

        # 将帧写入输出视频
        output_video.write(frame)
        cv2.imshow("1",frame)
        cv2.waitKey(1)

    # 释放视频对象
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()