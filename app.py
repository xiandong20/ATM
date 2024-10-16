#
# import gradio as gr
# import cv2
# import numpy as np
# import os
# from eval import eval
# from PIL import Image
#
# # 设置上传和结果文件夹
# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)
#
#
# import gradio as gr
#
#
# def greet(name):
#     return f"Hello, {name}!"
#
#
# with gr.Blocks() as demo:
#     with gr.Tabs():
#         with gr.Tab("Predict"):
#             with gr.Row():
#                 with gr.Column():
#                     name_input = gr.Textbox(label="Name")
#                     greet_button = gr.Button("Generator")
#                     output_text = gr.Textbox(label="Output")
#                     input_image=gr.Image(label="input Image")
#                 with gr.Column():
#                     output_image = gr.Image(label="input Image")
#
#         # with gr.Tab("Tab 2"):
#         #     image_input = gr.Image(label="Upload an Image")
#         #     image_output = gr.Image(label="Processed Image")
#
#     greet_button.click(greet, inputs=[name_input], outputs=[output_text])
#
# if __name__ == "__main__":
#     demo.launch()
#
# # python eval.py --device_target [Ascend] --device_id [0] --val_data_dir [./data/facades/test] --ckpt [./results/ckpt/Generator_200.ckpt] --pad_mode REFLECT
# # OR
# # bash scripts/run_eval_ascend.sh [DATASET_PATH] [DATASET_NAME] [CKPT_PATH] [RESULT_DIR]
#
#
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
    predict_path = eval(file_path)
    print(f"predict_path: {predict_path}")
    # 将处理后的图像转换为PIL图像
    result_image = Image.open(predict_path)
    return result_image

# 创建Gradio界面
with gr.Blocks(title="ATM变身器") as demo:
    # 主要功能区
    with gr.Row():
        gr.Markdown("# ATM变身器")
    with gr.Row():
        gr.Markdown("上传你心中的ATM并进行变身（上传一张ATM的铅笔画，然后点击提交）")

    with gr.Row():
        input_image = gr.Image(type="numpy", label="上传图像")
        output_image = gr.Image(type="pil", label="处理后的图像")

    # 处理按钮
    btn = gr.Button("提交")
    btn.click(process_image, inputs=input_image, outputs=output_image)
    with gr.Row():
        gr.Markdown("## 你可以把下面这些样例拖入上方输入区进行测试")
    # 在底部放置展示图
    with gr.Row():
        example1 = gr.Image(value="uploads/Samples/1.jpg", label="示例1")
        example2 = gr.Image(value="uploads/Samples/5.jpg", label="示例2")
        example3 = gr.Image(value="uploads/Samples/7.jpg", label="示例3")

# 启动Gradio应用
demo.launch()
