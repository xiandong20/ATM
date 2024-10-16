import gradio as gr
import cv2
import numpy as np
import os
from eval import eval
from PIL import Image

# 设置上传和结果文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 加载模型
# model = YOLO('yolov8n.pt')

def process_image(image):
    # 保存上传的图像
    filename = 'uploaded_image.jpg'
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(file_path, image)
    predict_path=eval(file_path)
    print(f"predict_path:{predict_path}")
    # 将处理后的图像转换为PIL图像
    result_image = Image.open(predict_path)
    return result_image

# 创建Gradio界面
iface = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="numpy", label="上传图像")],
    outputs=[gr.Image(type="pil", label="处理后的图像")],
    title="ATM变身器",
    description="上传你心中的ATM并进行变身（上传一张ATM的铅笔画，然后点击提交）"
                "你可以在这里下载一些样例并上传进行测试"
                ""
)

# 启动Gradio应用
iface.launch()

