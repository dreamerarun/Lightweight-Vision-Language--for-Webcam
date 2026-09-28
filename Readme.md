# Lightweight Vision-Language Webcam

A real-time Vision-Language Model (VLM) application that uses a webcam to understand and describe the surrounding environment.

The application runs locally using NVIDIA CUDA acceleration and is configured for an NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM.

It supports BLIP-Large, BLIP-2, and GIT-Large models and provides multiple interactive vision modes including image captioning, object understanding, detailed scene description, color analysis, and activity understanding.

---

## 📌 Overview

Traditional computer vision systems generally perform a specific task such as:

- Object detection
- Image classification
- Face detection
- Segmentation

Vision-Language Models provide a more flexible interface by combining visual information with language generation.

This project demonstrates a local webcam-based VLM pipeline:

```text
Webcam
   │
   ▼
Image Capture
   │
   ▼
Image Preprocessing
   │
   ▼
Vision-Language Model
   │
   ▼
Natural Language Output
   │
   ▼
OpenCV Visualization

The system continuously displays the webcam feed while periodically sending a frame to the selected VLM for analysis.

🚀 Features
Real-time webcam input
Local Vision-Language Model inference
NVIDIA CUDA acceleration
FP16 GPU inference
RTX 4050 support
Automatic CUDA detection
CPU fallback
Multiple pretrained VLMs
Image caption generation
Object-oriented scene understanding
Detailed scene description
Color understanding
Activity understanding
Live FPS display
Automatic inference every 3 seconds
Manual inference using SPACE
Screenshot capture
Interactive keyboard controls
GPU VRAM monitoring
OpenCV visualization overlay
🤖 Supported Models

The application supports three pretrained vision-language models.

Option	Model	Hugging Face Identifier	Main Purpose
1	BLIP-Large	Salesforce/blip-image-captioning-large	Image captioning
2	BLIP-2	Salesforce/blip2-opt-2.7b	Advanced vision-language understanding
3	GIT-Large	microsoft/git-large-coco	Image captioning
1. BLIP-Large

Model:

Salesforce/blip-image-captioning-large

BLIP-Large is the default model.

It is useful for general image captioning and provides a practical balance between model capability and GPU memory requirements.

Example output:

A person sitting in front of a computer.
2. BLIP-2

Model:

Salesforce/blip2-opt-2.7b

BLIP-2 provides more advanced vision-language capabilities.

It requires substantially more GPU memory than the other options and is therefore more demanding on a 6 GB RTX 4050.

3. GIT-Large

Model:

microsoft/git-large-coco

GIT-Large is another image-captioning model that can generate natural-language descriptions from images.

🖥️ Hardware Configuration

The application is designed for local NVIDIA GPU inference.

Target Configuration
GPU       : NVIDIA RTX 4050 Laptop GPU
VRAM      : 6 GB
RAM       : 16 GB
OS        : Ubuntu/Linux
Camera    : OpenCV-compatible webcam
CUDA      : NVIDIA CUDA

The application automatically checks whether CUDA is available before loading the selected model.

⚡ CUDA Configuration

The application performs a CUDA check during startup.

It checks:

PyTorch version
CUDA availability
CUDA version
GPU name
GPU VRAM

Example:

Checking CUDA setup...

PyTorch version: ...
CUDA available: True
CUDA version: ...
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
VRAM: 6.14 GB

If CUDA is unavailable, the application can fall back to CPU execution.

CPU inference can be considerably slower than GPU inference.

🔥 GPU Optimization

The application uses several optimizations for NVIDIA GPUs.

CUDA Device

The application targets the available NVIDIA GPU.

CUDA_VISIBLE_DEVICES=0
FP16

When CUDA is available, the model uses half precision:

model.half()

This reduces memory usage and can improve inference performance on compatible NVIDIA GPUs.

Inference Mode

Gradients are disabled during model generation:

torch.no_grad()

This avoids unnecessary training-related computation.

Evaluation Mode

The model is placed in evaluation mode:

model.eval()
📷 Webcam Processing

The webcam is configured to capture frames at:

640 × 480

Frames are captured using OpenCV.

The image is then:

BGR
 ↓
RGB
 ↓
PIL Image
 ↓
384 × 384
 ↓
VLM Processor
 ↓
GPU
 ↓
Generated Text

The image is resized to 384 × 384 before VLM processing to reduce computational requirements.

🔄 Complete Processing Pipeline
                 Webcam
                    │
                    ▼
             OpenCV Capture
                    │
                    ▼
              BGR Image
                    │
                    ▼
             RGB Conversion
                    │
                    ▼
              PIL Image
                    │
                    ▼
             384 × 384 Resize
                    │
                    ▼
           VLM Image Processor
                    │
                    ▼
              CUDA / FP16
                    │
                    ▼
       ┌─────────────────────────┐
       │ Vision-Language Model   │
       │                         │
       │ BLIP / BLIP-2 / GIT    │
       └─────────────────────────┘
                    │
                    ▼
          Natural Language Output
                    │
                    ▼
           OpenCV Display Overlay
👁️ Vision Modes

The application provides five vision modes.

1. Caption

The model generates a general description of the current webcam frame.

Example:

A person is sitting in front of a laptop.
2. Objects

The system asks:

What objects are in this image?

This mode is intended to provide a natural-language description of visible objects.

3. Details

The system asks:

Describe this image in detail

This mode attempts to generate a more descriptive explanation of the scene.

4. Colors

The system asks:

What colors are in this image?

This mode focuses on color information visible in the frame.

5. Activity

The system asks:

What is happening in this image?

This mode attempts to describe activities or actions occurring in the scene.

🎮 Keyboard Controls
Key	Function
q	Quit application
c	Cycle through modes
SPACE	Force immediate prediction
s	Save screenshot
1	Caption mode
2	Objects mode
3	Details mode
4	Colors mode
5	Activity mode
▶️ Running the Application

The main Python implementation is already included in this repository.

Run the existing Python file:

python <existing_python_file>.py

The program first performs a CUDA check.

It then displays:

Select Model:

1. BLIP-Large
2. BLIP-2
3. GIT-Large

Choose the required model.

Press:

1

for BLIP-Large.

Press:

2

for BLIP-2.

Press:

3

for GIT-Large.

Press Enter to use the default BLIP-Large model.

⏱️ Inference Frequency

The webcam feed runs continuously.

However, the VLM is not executed on every webcam frame.

The default prediction interval is:

3 seconds

This design reduces unnecessary GPU computation while maintaining a responsive webcam display.

The user can manually trigger a new prediction at any time using:

SPACE
📊 FPS Monitoring

The application calculates the webcam display FPS and displays it on the screen.

Example:

Mode: Caption                         FPS: 29.8

The FPS represents the webcam/display loop and should not be interpreted as the VLM inference speed.

The inference time is separately printed in the terminal.

Example:

Result (2.34s): A person is sitting in front of a laptop.
💾 Screenshot Capture

Press:

s

to save the current webcam frame.

The program creates a timestamp-based filename:

capture_<timestamp>.jpg

Example:

capture_1780000000.jpg
🖥️ User Interface

The OpenCV window contains three main areas.

Top Bar

Displays:

Current Mode
FPS
Main Area

Displays the live webcam feed.

Bottom Area

Displays:

AI Vision:
Generated description

and the keyboard controls.

Example:

Mode: Caption                         FPS: 29.8

AI Vision:
A person is sitting in front of a laptop.

q:Quit c:Mode SPACE:Now 1-5:Quick s:Save
🧪 CUDA Verification

Before running the project, NVIDIA GPU availability can be checked using:

nvidia-smi

PyTorch CUDA availability:

python -c "import torch; print(torch.cuda.is_available())"

GPU name:

python -c "import torch; print(torch.cuda.get_device_name(0))"

CUDA version reported by PyTorch:

python -c "import torch; print(torch.version.cuda)"

Expected result:

True

followed by the RTX 4050 GPU name.

🛠️ Troubleshooting
CUDA is not available

Run:

nvidia-smi

If the GPU is visible, check PyTorch:

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"

If:

torch.cuda.is_available()

returns:

False

the installed PyTorch environment may not have CUDA support or the NVIDIA software stack may require correction.

CUDA Out of Memory

The RTX 4050 Laptop GPU has 6 GB VRAM.

If the selected model exceeds available memory:

Close other GPU applications.
Check GPU memory:
nvidia-smi
Try BLIP-Large instead of BLIP-2.
Restart the Python process to release allocated GPU memory.

BLIP-2 is significantly more demanding than BLIP-Large.

Webcam Not Opening

Check available video devices:

ls /dev/video*

The default camera ID is:

0

If multiple cameras are connected, a different camera ID can be selected in the Python implementation.

🧰 Technologies

This project uses:

Python
PyTorch
CUDA
NVIDIA GPU
Hugging Face Transformers
OpenCV
NumPy
Pillow
BLIP
BLIP-2
GIT
🤖 Robotics Relevance

Although the current application is a standalone webcam VLM, the same architecture can be extended to robotics.

A robotics implementation could use:

Robot Camera
     │
     ▼
ROS 2 Image Topic
     │
     ▼
Vision-Language Model
     │
     ├── Scene Understanding
     ├── Object Information
     └── Activity Understanding
              │
              ▼
       Robot Decision Layer
              │
              ▼
          Robot Action

This creates a pathway from visual perception to language-based reasoning and eventually robot action.

🦾 Physical AI Extension

The project can serve as an early-stage perception component for Physical AI systems.

Potential extensions include:

Camera
  ↓
Vision-Language Model
  ↓
Scene Understanding
  ↓
Task Reasoning
  ↓
Action Planning
  ↓
Robot Controller

This can eventually be integrated with:

ROS 2
NVIDIA Isaac Sim
NVIDIA Isaac Lab
Robot manipulators
Mobile robots
Humanoid robots
Vision-Language-Action models
🚀 Future Improvements

The following features can be added in future versions:

 ROS 2 camera integration
 ROS 2 VLM node
 YOLO object detection
 SAM/SAM2 segmentation
 Voice input
 Text-to-speech
 Conversational VLM interaction
 Multiple camera support
 NVIDIA Isaac Sim integration
 Robot camera integration
 Jetson deployment
 TensorRT optimization
 Asynchronous VLM inference
 Vision-Language-Action integration
 Robot manipulation integration
📈 Possible Applications

The system can be used for experimentation in:

Computer Vision
Vision-Language Models
Robotics Perception
Human-Robot Interaction
Physical AI
Autonomous Systems
Smart Camera Systems
Robot Learning
Vision-Language-Action Research
📁 Repository Structure

The repository intentionally keeps the implementation separate from the documentation.

repository/
│
├── README.md
│
└── <existing Python source file>

The Python source code is already provided in the repository and does not need to be duplicated inside this README.

🔬 Project Objective

The main objective is to demonstrate that a pretrained Vision-Language Model can be deployed locally on a consumer NVIDIA GPU and used for real-time webcam-based scene understanding.

The project also provides a starting point for connecting VLM-based visual perception with robotics and Physical AI systems.

👨‍💻 Author

Arun M.

B.Tech Robotics & Automation

Areas of interest:

Robotics
Physical AI
Vision-Language Models
Computer Vision
Robot Learning
ROS 2
NVIDIA Isaac Sim
Autonomous Systems
Vision-Language-Action Models
⭐ Summary

This project demonstrates a complete local VLM webcam pipeline:

Webcam
   ↓
Image Processing
   ↓
Vision-Language Model
   ↓
Natural Language Understanding
   ↓
Real-Time Visualization
