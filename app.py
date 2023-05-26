import pandas as pd
import gradio as gr
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel

device="cpu"
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
cat_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
cap_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)

def predict(image, max_length=64, num_beams=4):
    image = image.convert('RGB')
    image = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
    caption_ids = cap_model.generate(image, max_length=max_length)[0]
    caption_text = clean_text(cat_tokenizer.decode(caption_ids))
    return caption_text

input = gr.components.Image(label="Upload Image", type = 'pil')
caption = gr.components.Textbox(type="text", label="Captions")
examples = [f"e{i}.jpg" for i in range(1,7)]

title = "Image Caption"
description = "Made by: Swapnil Tripathi"

interface = gr.Interface(
    fn=predict,
    description=description,
    inputs=input,
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.slate
    ),
    outputs=caption,
    examples=examples,
    title=title,
)

interface.launch(debug=True, share=True)
