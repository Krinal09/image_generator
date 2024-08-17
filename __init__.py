# from flask import Flask, render_template, request, send_file
# import torch
# from diffusers import StableDiffusionXLPipeline
# import os
# from concurrent.futures import ThreadPoolExecutor

# app = Flask(__name__)

# # Set up device: use GPU if available, otherwise CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# try:
#     # Load the Stable Diffusion XL Base 1.0 model only once
#     pipe = StableDiffusionXLPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-xl-base-1.0", 
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
#     )
#     pipe.to(device)
#     pipe.safety_checker = None  # Disable safety checker for performance
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()

# # Ensure the folder for storing generated images exists
# IMAGE_FOLDER = 'static'
# os.makedirs(IMAGE_FOLDER, exist_ok=True)

# def generate_single_image(i, prompt, num_steps, guidance):
#     """Generate a single image asynchronously."""
#     try:
#         if pipe.tokenizer is None:
#             raise ValueError("Pipeline's tokenizer not initialized")
        
#         image = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance).images[0]
#         image_path = os.path.join(IMAGE_FOLDER, f'generated_image_{i}.png')
#         image.save(image_path)
#         # return f'generated_image_{i}.png'
#     except Exception as e:
#         print(f"Error generating image {i}: {e}")
#         return None

# def generate_images(prompt, num_steps=30, guidance=7.5):
#     """Generate six images in parallel with a ThreadPoolExecutor."""
#     image_urls = []
#     with ThreadPoolExecutor(max_workers=3) as executor:  # Limit concurrency for stability
#         futures = [executor.submit(generate_single_image, i, prompt, num_steps, guidance) for i in range(6)]
#         image_urls = [future.result() for future in futures if future.result() is not None]
#     return image_urls

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     image_urls = []
    
#     if request.method == 'POST':
#         prompt = request.form['prompt']
#         if prompt:
#             image_urls = generate_images(prompt)
    
#     return render_template('index.html', image_urls=image_urls)

# @app.route('/download/<filename>')
# def download_image(filename):
#     path = os.path.join(IMAGE_FOLDER, filename)
#     return send_file(path, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True)









from flask import Flask, render_template, request, send_file
import torch
from diffusers import StableDiffusionPipeline
import os

app = Flask(__name__)

# Set up device: use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    # Load the Stable Diffusion v1-4 model
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe.to(device)
    pipe.safety_checker = None  # Disable safety checker for performance
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Ensure the folder for storing generated images exists
IMAGE_FOLDER = 'static'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

def generate_single_image(i, prompt, num_steps, guidance):
    """Generate a single image synchronously."""
    try:
        # Generate the image
        image = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance).images[0]
        image_path = os.path.join(IMAGE_FOLDER, f'generated_image_{i}.png')
        image.save(image_path)
        return f'generated_image_{i}.png'
    except Exception as e:
        print(f"Error generating image {i}: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    image_urls = []
    
    if request.method == 'POST':
        prompt = request.form['prompt']
        if prompt:
            # Generate images sequentially for debugging purposes
            for i in range(6):
                image_url = generate_single_image(i, prompt, num_steps=100, guidance=7.5)
                if image_url:
                    image_urls.append(image_url)
    
    return render_template('index.html', image_urls=image_urls)

@app.route('/download/<filename>')
def download_image(filename):
    path = os.path.join(IMAGE_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
