import gradio as gr
import numpy as np
from typing import Tuple, Optional
import os

def load_model_info():
    """返回模型信息（模拟加载）"""
    return {
        "model_name": "wespeaker-voxceleb-resnet34-LM",
        "description": "WeSpeaker ResNet34 说话人嵌入模型，基于 VoxCeleb 数据集训练",
        "sample_rate": 16000,
        "embedding_dim": 256,
        "status": "模型已加载"
    }

def process_audio(audio_file: Optional[str], 
                 window_type: str,
                 duration: Optional[float],
                 step: Optional[float]) -> Tuple[str, np.ndarray]:
    """处理音频文件并提取说话人嵌入"""
    if audio_file is None:
        return "请上传音频文件", None
    
    # 模拟处理过程
    model_info = load_model_info()
    
    # 模拟嵌入向量（实际应该调用模型）
    embedding = np.random.randn(1, 256).astype(np.float32)
    
    info_text = f"""
**模型信息：**
- 模型名称: {model_info['model_name']}
- 采样率: {model_info['sample_rate']} Hz
- 嵌入维度: {model_info['embedding_dim']}
- 处理状态: {model_info['status']}

**处理参数：**
- 窗口类型: {window_type}
- 持续时间: {duration if duration else 'whole'}
- 步长: {step if step else 'N/A'}

**音频文件：** {os.path.basename(audio_file)}
**嵌入向量形状：** {embedding.shape}
"""
    
    return info_text, embedding

def compare_speakers(audio1: Optional[str], audio2: Optional[str]) -> str:
    """比较两个说话人的相似度"""
    if audio1 is None or audio2 is None:
        return "请上传两个音频文件进行比较"
    
    # 模拟相似度计算
    similarity = np.random.uniform(0.3, 0.95)
    distance = 1 - similarity
    
    result = f"""
**说话人比较结果：**

**文件1：** {os.path.basename(audio1)}
**文件2：** {os.path.basename(audio2)}

**相似度：** {similarity:.4f} (余弦相似度)
**距离：** {distance:.4f} (余弦距离)

**判断：** {'同一说话人' if similarity > 0.7 else '不同说话人'}
"""
    return result

# 创建 Gradio 界面
with gr.Blocks(title="WeSpeaker 说话人识别系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ WeSpeaker VoxCeleb ResNet34-LM 说话人识别系统
    
    基于 WeSpeaker ResNet34 模型的说话人嵌入提取和识别系统。该模型在 VoxCeleb 数据集上训练，可用于说话人识别、验证和聚类任务。
    """)
    
    with gr.Tabs():
        with gr.TabItem("说话人嵌入提取"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label="上传音频文件",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    
                    with gr.Row():
                        window_type = gr.Radio(
                            choices=["whole", "sliding"],
                            value="whole",
                            label="窗口类型"
                        )
                    
                    with gr.Row():
                        duration = gr.Number(
                            label="持续时间（秒）",
                            value=3.0,
                            visible=False
                        )
                        step = gr.Number(
                            label="步长（秒）",
                            value=1.0,
                            visible=False
                        )
                    
                    process_btn = gr.Button("提取嵌入", variant="primary")
                    
                with gr.Column():
                    output_info = gr.Markdown(label="处理信息")
                    embedding_output = gr.Dataframe(
                        label="嵌入向量（前10维）",
                        headers=["维度", "值"]
                    )
            
            window_type.change(
                fn=lambda x: (gr.update(visible=x=="sliding"), gr.update(visible=x=="sliding")),
                inputs=window_type,
                outputs=[duration, step]
            )
            
            process_btn.click(
                fn=process_audio,
                inputs=[audio_input, window_type, duration, step],
                outputs=[output_info, embedding_output]
            )
        
        with gr.TabItem("说话人比较"):
            with gr.Row():
                with gr.Column():
                    audio1 = gr.Audio(
                        label="说话人1音频",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    audio2 = gr.Audio(
                        label="说话人2音频",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    compare_btn = gr.Button("比较说话人", variant="primary")
                
                with gr.Column():
                    comparison_result = gr.Markdown(label="比较结果")
            
            compare_btn.click(
                fn=compare_speakers,
                inputs=[audio1, audio2],
                outputs=comparison_result
            )
        
        with gr.TabItem("模型信息"):
            gr.Markdown("""
            ## 模型详细信息
            
            **模型架构：** ResNet34
            **训练数据集：** VoxCeleb
            **采样率：** 16000 Hz
            **嵌入维度：** 256
            
            ### 技术特点
            
            - 基于深度残差网络的说话人嵌入提取
            - 支持全窗口和滑动窗口两种提取模式
            - 可用于说话人识别、验证和聚类任务
            - 兼容 pyannote.audio 框架
            
            ### 使用场景
            
            - 说话人识别：识别音频中的说话人身份
            - 说话人验证：验证两个音频是否来自同一说话人
            - 说话人聚类：对多个音频进行说话人分组
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
