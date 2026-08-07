import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image

class PhotoRestorationPipeline:
    def __init__(self, model_path='yolov8n.pt', color_model_path='deoldify_generator.pth', face_model_path='gfpgan.pth'):
        """
        Initializes the pipeline and loads the pre-trained YOLO model into memory.
        Note: We are using the default 'yolov8n.pt' here for demonstration. 
        In a real scenario, you would train a custom YOLOv8 model on a dataset 
        of scratched photos and load your custom 'best.pt' file here.
        """
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(model_path):
            model_path = os.path.join(base_dir, model_path)
            
        print("Loading YOLO model...")
        self.detector = YOLO(model_path)
        print("Model loaded successfully.")

        # Setup PyTorch Device (Use GPU if available for faster deep learning processing)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PyTorch using device: {self.device}")
        
        # Initialize Colorization Model
        self.color_model = self._load_color_model(color_model_path)
        
        # Initialize Face Enhancement Model (Step 4)
        self.face_model = self._load_face_model(face_model_path)

    def _load_face_model(self, path):
        """
        Loads the PyTorch face enhancement model (e.g., GFPGAN or CodeFormer).
        """
        print("Loading Face Enhancement model...")
        try:
            # Here is where you plug in the actual GFPGAN architecture.
            # Example implementation using the official GFPGANer package:
            # from gfpgan import GFPGANer
            # restorer = GFPGANer(model_path=path, upscale=2, arch='clean', channel_multiplier=2, device=self.device)
            # return restorer
            
            print("Note: PyTorch face enhancement hook initialized (Waiting for model weights).")
            return None 
        except Exception as e:
            print(f"Warning: Could not load face model: {e}")
            return None

    def _load_color_model(self, path):
        """
        Loads the OpenCV DNN Colorization model, automatically downloading missing files.
        """
        print("Loading Deep Colorization model...")
        try:
            import os
            import requests
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "models")
            
            # 1. Create the 'models' directory if it doesn't exist
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                print("Created 'models' directory.")
                
            prototxt = os.path.join(models_dir, "colorization_deploy_v2.prototxt")
            caffemodel = os.path.join(models_dir, "colorization_release_v2.caffemodel")
            npy_file = os.path.join(models_dir, "pts_in_hull.npy")
            
            # 2. Define direct download links for the required Caffe files
            # 2. Define direct download links for the required Caffe files
            urls = {
                prototxt: "https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt",
                caffemodel: "https://www.dropbox.com/s/dx0qvhhx5huu46s/colorization_release_v2.caffemodel?dl=1",
                npy_file: "https://github.com/richzhang/colorization/raw/caffe/resources/pts_in_hull.npy"
            }
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            
            # 3. Clean up corrupted or incomplete previous downloads
            for file_path, url in urls.items():
                if os.path.exists(file_path):
                    file_size_kb = os.path.getsize(file_path) / 1024
                    # caffemodel should be ~125MB (~128,000 KB). If < 1000 KB or 0 KB, delete it.
                    if "caffemodel" in file_path and file_size_kb < 1000:
                        print(f"Corrupted model file detected ({file_size_kb:.1f} KB). Deleting...")
                        os.remove(file_path)
                    elif file_size_kb == 0:
                        os.remove(file_path)

                # Download file if missing
                if not os.path.exists(file_path):
                    print(f"Downloading {os.path.basename(file_path)}... (This may take a minute)")
                    response = requests.get(url, headers=headers, stream=True)
                    response.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
            
            # 4. Load the network using OpenCV 
            import cv2
            import numpy as np
            net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            pts_in_hull = np.load(npy_file)
            
            pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
            net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
            net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]
            print("Colorization models loaded successfully.")
            return net
            
        except Exception as e:
            print(f"Warning: Could not load color model: {e}")
            return None

    def generate_inpainting_mask(self, original_image, confidence_threshold=0.25):
        """
        Step 1: Detects defects and generates a binary mask (Prototype using OpenCV).
        """
        print("Detecting scratches using OpenCV Top-Hat morphology...")
        # 1. Convert to grayscale
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Top-Hat Transform to isolate bright features (scratches) against the background
        # We use a relatively large kernel to catch thick scratches
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # 3. Threshold the tophat image to create a binary mask
        # Since the background lighting is removed by tophat, we can use a lower threshold
        _, mask = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)
        
        # 4. Morphological dilation to expand the mask slightly and ensure edges are covered
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        
        # We run the YOLO detector just so the Streamlit UI has results to draw,
        # but we ignore its mask output to fix the "people smearing" issue.
        results = self.detector(original_image, conf=confidence_threshold)
        
        return mask, results

    def apply_inpainting(self, original_image, mask, inpaint_radius=3):
        """
        Step 2: Uses the generated mask to repair the structural damage in the image.
        """
        print("Applying OpenCV Inpainting (Navier-Stokes algorithm)...")
        # cv2.INPAINT_NS (Navier-Stokes) is generally better for structural repairs
        # cv2.INPAINT_TELEA is another option that is sometimes faster
        restored_image = cv2.inpaint(original_image, mask, inpaint_radius, cv2.INPAINT_NS)
        return restored_image

    def apply_colorization(self, restored_image):
        """
        Step 3: Converts the repaired grayscale image to color using Deep Learning.
        """
        if self.color_model is None:
            print("Skipping Step 3: Colorization weights not found.")
            return restored_image
            
        print("Applying OpenCV Deep Colorization...")
        
        # 1. Prepare Data
        # Scale to 0-1 range and convert to LAB color space
        img_rgb = cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype("float32") / 255.0
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        
        # 2. Resize L channel for network input (224x224) and subtract mean (50)
        L = img_lab[:, :, 0]
        L_resized = cv2.resize(L, (224, 224))
        L_resized -= 50
        
        # 3. Model Inference
        self.color_model.setInput(cv2.dnn.blobFromImage(L_resized))
        ab_predicted = self.color_model.forward()[0, :, :, :].transpose((1, 2, 0))
        
        # 4. Post-Process Data
        # Resize predicted ab channels back to original image size
        original_h, original_w = restored_image.shape[:2]
        ab_predicted = cv2.resize(ab_predicted, (original_w, original_h))
        
        # Combine the original L channel with the predicted ab channels
        colorized_lab = np.concatenate((L[:, :, np.newaxis], ab_predicted), axis=2)
        
        # Convert back to BGR 0-255 format
        colorized_rgb = cv2.cvtColor(colorized_lab, cv2.COLOR_Lab2RGB)
        colorized_bgr = cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2BGR)
        final_bgr = np.clip(colorized_bgr * 255, 0, 255).astype(np.uint8)
        
        return final_bgr

    def apply_face_enhancement(self, colorized_image):
        """
        Step 4: Uses a face restoration model to sharpen and enhance (Simulated Unsharp Mask).
        """
        print("Applying Simulated Face Enhancement (Unsharp Mask)...")
        
        # We simulate the GFPGAN sharpening effect by applying an unsharp mask
        # 1. Blur the image
        gaussian = cv2.GaussianBlur(colorized_image, (9, 9), 10.0)
        
        # 2. Add the weighted difference back to the original to sharpen it
        enhanced_img = cv2.addWeighted(colorized_image, 1.5, gaussian, -0.5, 0)
        
        return enhanced_img

    def process_image(self, image_path, output_path="restored_output.jpg"):
        """
        Master function to run the image through all 4 steps sequentially.
        """
        # Load the damaged image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}. Check the file path.")

        # Step 1: Detect and Mask
        print("Detecting damage and building mask...")
        mask, _ = self.generate_inpainting_mask(img)

        # Step 2: Repair
        restored_img = self.apply_inpainting(img, mask)

        # Step 3: Colorization
        colorized_img = self.apply_colorization(restored_img)
        
        # Step 4: Face Enhancement & Super-Resolution
        final_img = self.apply_face_enhancement(colorized_img)

        # Save the results so you can see what happened under the hood
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mask_path = os.path.join(script_dir, "generated_mask.jpg")
        final_out_path = os.path.join(script_dir, output_path)
        
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(final_out_path, final_img)
        
        print(f"Success! Saved mask to '{mask_path}' and final restored image to '{final_out_path}'.")
        return final_img

# ==========================================
# How to use the pipeline
# ==========================================
if __name__ == "__main__":
    # Initialize our pipeline
    pipeline = PhotoRestorationPipeline()
    
    # Run a test image through the pipeline
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(script_dir, "damaged_photo.jpg")
    
    try:
        pipeline.process_image(test_image_path, "restored_output.jpg")
    except Exception as e:
        print(f"Error: {e}")
        print("Please provide a valid image path to test the pipeline.")