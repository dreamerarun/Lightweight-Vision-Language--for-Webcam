"""
Fixed Lightweight Vision-Language Model for Webcam
Properly configured for RTX 4050 with CUDA fixes
"""

import torch
import cv2
import numpy as np
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, Blip2ForConditionalGeneration,
    AutoModelForCausalLM
)
from PIL import Image
import time
from collections import deque
import os

class LightweightVLWebcam:
    def __init__(self, model_choice="blip", device="cuda"):
        """Initialize vision-language model with proper CUDA setup"""
        
        # Force CUDA environment variables
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        print(f"Loading {model_choice} model...")
        self.device = device
        self.model_choice = model_choice
        
        # Check CUDA again
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available!")
            print("Falling back to CPU - will be slow")
            device = "cpu"
            self.device = "cpu"
        
        try:
            if model_choice == "blip":
                # BLIP-Base - most reliable
                self.processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-large"
                )
                self.model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-large"
                )
                
            elif model_choice == "blip2":
                # BLIP-2 - better quality
                self.processor = AutoProcessor.from_pretrained(
                    "Salesforce/blip2-opt-2.7b"
                )
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                
            elif model_choice == "git":
                # GIT - fast and accurate
                self.processor = AutoProcessor.from_pretrained(
                    "microsoft/git-large-coco"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/git-large-coco"
                )
            
            # Move to device
            if device == "cuda":
                self.model = self.model.half()  # Use FP16 for speed
            self.model = self.model.to(device)
            self.model.eval()
            
            print(f"✓ {model_choice} model loaded on {device}!")
            
            # Show memory usage
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1e9
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✓ VRAM Usage: {vram_used:.2f}GB / {vram_total:.2f}GB")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, frame, prompt=None):
        """Generate description from frame"""
        try:
            # Convert to PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Resize for faster processing
            pil_image = pil_image.resize((384, 384))
            
            if self.model_choice == "blip":
                # BLIP captioning (no prompt needed)
                inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=3,
                        min_length=10,
                        top_p=0.9,
                        repetition_penalty=1.5
                    )
                
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                
                # If we have a prompt, do VQA
                if prompt and "what" in prompt.lower():
                    inputs = self.processor(
                        pil_image, 
                        text=prompt,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_length=50)
                    
                    answer = self.processor.decode(out[0], skip_special_tokens=True)
                    return answer
                    
                return caption
                
            elif self.model_choice == "blip2":
                # BLIP-2
                if prompt:
                    inputs = self.processor(
                        pil_image,
                        text=prompt,
                        return_tensors="pt"
                    ).to(self.device, torch.float16)
                else:
                    inputs = self.processor(
                        pil_image,
                        return_tensors="pt"
                    ).to(self.device, torch.float16)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=50,
                        min_length=10
                    )
                
                result = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].strip()
                
                return result
                
            elif self.model_choice == "git":
                # GIT
                inputs = self.processor(
                    images=pil_image,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values=inputs.pixel_values,
                        max_length=50
                    )
                
                result = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
                
                return result
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def run_webcam(self, camera_id=0):
        """Run webcam demo"""
        print("\n" + "="*60)
        print("Vision-Language Webcam Demo")
        print("="*60)
        print("Controls:")
        print("  q - Quit")
        print("  c - Cycle through modes")
        print("  s - Save screenshot")
        print("  SPACE - Force new prediction")
        print("  1-5 - Quick prompts")
        print("="*60 + "\n")
        
        # Modes
        modes = [
            ("Caption", None),  # Just describe
            ("Objects", "What objects are in this image?"),
            ("Details", "Describe this image in detail"),
            ("Colors", "What colors are in this image?"),
            ("Activity", "What is happening in this image?")
        ]
        current_mode = 0
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        last_prediction = "Initializing..."
        prediction_interval = 3.0  # Every 3 seconds
        last_time = 0
        fps_buffer = deque(maxlen=30)
        
        mode_name, prompt = modes[current_mode]
        print(f"Mode: {mode_name}")
        if prompt:
            print(f"Prompt: {prompt}")
        
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Auto predict
            current_time = time.time()
            if current_time - last_time >= prediction_interval:
                mode_name, prompt = modes[current_mode]
                print(f"\n[{mode_name}] Analyzing frame...")
                start_pred = time.time()
                last_prediction = self.predict(frame, prompt)
                pred_time = time.time() - start_pred
                print(f"Result ({pred_time:.2f}s): {last_prediction}")
                last_time = current_time
            
            # FPS
            fps = 1.0 / max(0.001, time.time() - frame_start)
            fps_buffer.append(fps)
            avg_fps = np.mean(fps_buffer)
            
            # Draw
            self.draw_overlay(frame, mode_name, last_prediction, avg_fps)
            
            cv2.imshow('VL Webcam', frame)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                current_mode = (current_mode + 1) % len(modes)
                mode_name, prompt = modes[current_mode]
                print(f"\n>>> Changed to mode: {mode_name}")
                if prompt:
                    print(f"    Prompt: {prompt}")
            elif key == ord(' '):
                mode_name, prompt = modes[current_mode]
                print(f"\n[MANUAL] Analyzing...")
                start_pred = time.time()
                last_prediction = self.predict(frame, prompt)
                pred_time = time.time() - start_pred
                print(f"Result ({pred_time:.2f}s): {last_prediction}")
                last_time = current_time
            elif key == ord('s'):
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            elif ord('1') <= key <= ord('5'):
                idx = key - ord('1')
                if idx < len(modes):
                    current_mode = idx
                    mode_name, prompt = modes[current_mode]
                    print(f"\n>>> Quick select: {mode_name}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nCleaning up...")
    
    def draw_overlay(self, frame, mode, prediction, fps):
        """Draw overlay"""
        h, w = frame.shape[:2]
        
        # Top bar - mode and FPS
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"Mode: {mode}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Bottom bar - prediction
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "AI Vision:", (10, h-75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Wrap text
        max_chars = 80
        lines = []
        words = prediction.split()
        current_line = []
        current_len = 0
        
        for word in words:
            if current_len + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_len += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_len = len(word)
        if current_line:
            lines.append(' '.join(current_line))
        
        y_offset = h - 50
        for i, line in enumerate(lines[:2]):  # Max 2 lines
            cv2.putText(frame, line, (10, y_offset + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        # Controls
        cv2.putText(frame, "q:Quit c:Mode SPACE:Now 1-5:Quick s:Save",
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)


def fix_cuda():
    """Try to fix CUDA issues"""
    print("Checking CUDA setup...")
    
    # Check CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("\n⚠️  CUDA NOT AVAILABLE!")
        print("This usually means:")
        print("1. PyTorch not installed with CUDA support")
        print("2. CUDA environment variables not set")
        print("\nTo fix:")
        print("pip uninstall torch torchvision torchaudio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False


def main():
    print("\n" + "="*60)
    print("Vision-Language Webcam Demo - FIXED VERSION")
    print("="*60 + "\n")
    
    # Fix CUDA first
    has_cuda = fix_cuda()
    
    if not has_cuda:
        response = input("\nCUDA not working. Continue with CPU (SLOW)? [y/N]: ")
        if response.lower() != 'y':
            print("Exiting. Please fix CUDA installation.")
            return
    
    print("\n" + "="*60)
    print("Select Model:")
    print("="*60)
    print("1. BLIP-Large (Best quality, ~2GB)")
    print("2. BLIP-2 (Better, ~4GB, needs GPU)")
    print("3. GIT-Large (Fast, ~2GB)")
    print("="*60)
    
    choice = input("Enter choice (1-3) or press Enter for [1]: ").strip()
    
    model_map = {
        "1": "blip",
        "2": "blip2",
        "3": "git",
        "": "blip"
    }
    
    model_choice = model_map.get(choice, "blip")
    device = "cuda" if has_cuda else "cpu"
    
    print(f"\nLoading {model_choice} on {device}...")
    
    try:
        demo = LightweightVLWebcam(model_choice=model_choice, device=device)
        demo.run_webcam()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDemo ended!")


if __name__ == "__main__":
    main()

