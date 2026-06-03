import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import warnings
warnings.filterwarnings("ignore")

##Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_caption(image):
    raw_image = Image.open(image).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

## build the Gradio interface
interface = gr.Interface(fn=generate_caption, inputs=gr.Image(type="filepath"), outputs= gr.Textbox(label= "Generated Caption"), title="Image Captioning with BLIP", description="Upload an image to generate a caption using the BLIP model.")
## launch the interface
if __name__ == "__main__":
    interface.launch()