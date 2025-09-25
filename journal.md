# 🌸💖 Error Journal — Aditri Singh 💖🌸  

**🕒 Date & Time Started:** 27 Aug 2025, 07:15 PM  

---

# 🚨 Error Log Entry  

## 🔹 Error Title  
Problem when testing onnx model-NMS

### ❌ Error Message  
```bash
Desktop\Drone\runs\detect\train4\weights\best.onnx --img C:\Users\Aditri\Desktop\Drone\00027.jpg --imgsz 640
>>
Input name: images
Inference time: 48.6 ms
output[0] shape: (1, 6, 8400), min=0.000000, max=634.357605
Saved out.jpg — open it to check detections
(drone) PS C:\Users\Aditri\Desktop\Drone> 
````

### 📝 Description
I made test_onnx_infer.py to chcek if my exported onnx model is runnig corrcetly i tried it on an imge and this came


### 🔍 Debugging Process

I just knew from first glance its an NMS problem
### ✅ Solution

WRONG CODE-
import numpy as np

boxes = [[10, 20, 50, 60], [12, 22, 48, 58], [200, 200, 250, 250]]  # two overlapping, one far
scores = [0.9, 0.8, 0.7]

# ❌ Wrong: loop compares with itself (fragile and messy)
final_boxes = []
while boxes:
    box = boxes[0]     # pick first box
    score = scores[0]
    final_boxes.append((box, score))
    
    # comparing with ALL boxes (including itself!)
    remove_idx = []
    for i, b in enumerate(boxes):
        # if overlap > threshold → remove
        if iou(box, b) > 0.5:   # <- includes itself!
            remove_idx.append(i)

    # remove overlaps
    boxes = [b for i, b in enumerate(boxes) if i not in remove_idx]
    scores = [s for i, s in enumerate(scores) if i not in remove_idx]

print(final_boxes)





#why wrong?
👉 Problem: the first box overlaps 100% with itself, so it deletes itself. Result = empty or wrong boxes.


✅ The Fix with Slice [1:]


# Instead of comparing with ALL boxes, skip the first (itself)
for b in boxes[1:]:
    if iou(box, b) > 0.5:
        # mark for removal
👉 But this is still fragile (hard to maintain, can break if indexing changes).

# Final fix (code/command/snippet)

⭐ Breakthrough: Use OpenCV’s Built-in NMS
import cv2

boxes = [[10, 20, 50, 60], [12, 22, 48, 58], [200, 200, 250, 250]]
scores = [0.9, 0.8, 0.7]

# OpenCV needs format: [x, y, w, h] instead of [x1, y1, x2, y2]
boxes_xywh = []
for x1, y1, x2, y2 in boxes:
    boxes_xywh.append([x1, y1, x2-x1, y2-y1])

# Apply NMS
indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.5, nms_threshold=0.4)

final_boxes = [boxes[i] for i in indices.flatten()]
print(final_boxes)


👉 OpenCV handles everything (no risk of comparing a box with itself). Clean, reliable, and easy.









## Error Title  
Problem: docker not found inside WSL2

hat happened: Initially......I tried running -docker run ... inside Ubuntu WSL2, but it said docker: command not found.

## WHY ERROR HAPPENED?
DOCKER wasnt integrated with wsl2 yet so i opend docker desktop ->settings->resources->WSL INTEGRATION


## Error Title
Permission denied(Docker)



## WHAT TERMINAL SAID
permission denied while trying to connect to the Docker daemon


# # What happened
After installing Docker, when I ran:

docker run hello-world


## I GOT THIS ERROR
permission denied while trying to connect to the Docker daemon socket



## REASON I GOT THIS ERRROR
Reason: By default, only the root user can access the Docker daemon (/var/run/docker.sock)


## Temporary Fix:
I used sudo docker run hello-world → ✅ it worked, but running everything with sudo is inconvenient :(

## PERMANENT FIX:
Allow the(us) user to use Docker without sudo

## STEPS HOW I DID PERMANENT FIXING:
->CREATED DOCKER GROUP
->ADDED MY USER TO DOCKER GROUP


I ran this-
sudo groupadd docker       # (created docker group if not already present)
sudo usermod -aG docker $USER   # (added your user to docker group)
newgrp docker              # (refreshed your session so group change takes effect)



Verified with:
docker run hello-world


->WORKED PERFECTLY!!! :D






## FEW THINGS I GOT TO KNOW:
1-Docker requires a daemon (background service) that runs with root privileges.

2-Regular users can’t access it unless they belong to the docker group.

3-The newgrp docker command is important — it refreshes the shell session to apply group changes immediately.

4-Running hello-world is the classic way to confirm Docker is installed & working.




## 🌊🌊 ⚓🏴‍☠️🧜 MY ADVENTOUROUS DOCKER PULL JOURNEY 🐚🪸🌊🌊

## 1) FIRST ATTEMPT

docker pull nvcr.io/nvidia/tensorrt:24.07-py3

->Docker started pulling layers (big chunks of the TensorRT image)
->One huge layer (aa00ff708333) started downloading (size ~4.13GB)

## ERROR I GOT

short read: expected 4130015270 bytes but got ... unexpected EOF


👉Meaning the download was cut off midway (network drop or timeout).

## 2) I CHECKED THE DISK SPACE
 df -h

👉 Had PLENTYYYY of storage 

## 3) TRIED REMOVING PARTIAL IMAGE

docker rmi nvcr.io/nvidia/tensorrt:24.07-py3


## DOCKER SAID-?
No such image

## why?
->THATS BECAUSE IMAGE WAS NEVER PULLED 😭😭😭
the earlier pull never completed successfully, so there was nothing to remove LMAOOO


## 4) LOGIN CONFUSION 😭🙏
docker login nvcr.io

->I first tried with my username/password → ❌ unauthorized

THEN I GOT TO KNOW...
👉NVIDIA’s NGC registry actually doesn’t use normal usernames — it uses an API key.

Correct login is:-
Username: $oauthtoken
Password: <your API key>


## 5) SECOND PULL ATTEMPT 😭🙏
docker pull nvcr.io/nvidia/tensorrt:24.07-py3

Again started downloading…

❌ But failed with unexpected EOF (same issue, layer download cut short)

## 6) TRIED WITH SUDO
sudo docker pull nvcr.io/nvidia/tensorrt:24.07-py3

👉Same image, but now with sudo
->This didn’t fix the root issue (network instability), but it retried download


## 7)Finally succeeded 😭🎉

After repeated retries, Docker cached completed layers and eventually finished the big ~4GB layer

BASH---
Digest: sha256:...
Status: Downloaded newer image for nvcr.io/nvidia/tensorrt:24.07-py3

FINALLY officially through the “Docker Pull Boss Fight” 🙏🙏🙏



## THINGS I GOT TO LEARN AFTER THIS FIGHT
Docker images = multiple layers; if a big layer fails mid-download, you can retry and it resumes from cached progress.

unexpected EOF = download was cut off early (network/disk hiccup).

docker rmi only removes fully downloaded images, not failed/corrupted pulls.

To access NVIDIA’s registry (nvcr.io), you must log in with API key:

Username = $oauthtoken

Password = your NGC API key

Persistence is key: each retry pulls missing layers until everything is complete.

Once Status: Downloaded newer image appears, the image is ready to use.



## Making WSL2 Docker Talk to GPU ⚓🚢

## 1) Installing NVIDIA GPU Driver (Windows side)
I installed the NVIDIA GPU driver for my RTX card on Windows.

Without this, my GPU is just sitting there and Docker/WSL2 can’t see it.

This is like giving my Windows PC the ability to “talk” to the GPU.

👉 Example: In Windows Device Manager, my RTX 4060 shows up after driver install



## 2) Installing CUDA Toolkit + NVIDIA Container Toolkit (WSL2 side)
Inside WSL2 (Ubuntu), I installed CUDA Toolkit and nvidia-container-toolkit.

This step is important because Docker containers don’t magically know about GPUs.

These toolkits act like a bridge that connects WSL2 and Docker to the GPU driver on Windows.

👉 Commands I ran (inside Ubuntu WSL2):
## 3) Install NVIDIA container runtime
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker


## 4) Testing GPU in WSL2

I tested if WSL2 can see my GPU with:

nvidia-smi


If this shows my RTX 4060 and driver info → 🎉 success.

This was the “aha!” moment because it meant the GPU was exposed to WSL2.


## 5) Running Docker with GPU Access
Finally, I ran a test Docker container to check GPU access:

docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi


Output showed the same GPU info inside the container.

That proved Windows driver → WSL2 → Docker → GPU chain was working


✅ End result:
Now my TensorRT FP16 engine inside Docker can use the GPU instead of CPU




## 😭🙏🙏I HAD TO SHUT DOWN DOCKER AND WSL2 WINDOW CAUSE I am ADDING  tracking pipeline using YOLOv8n for detection and OSNet for Re-Identification (Re-ID). For cross-platform inference and GPU acceleration, I needed OSNet in ONNX format ,WE'LL CONTINUE WITH DOCKER AN DWSL2 LATER 


#MAKING FP16 ENGINE OF YOLOV8
I faced some trpouble whne making an fp16 engine of my yolo 
This-
(drone-env) aditrisingh@LAPTOP-6T8704AT:/mnt/c/Users/Aditri/Desktop/Drone$ python3 export_dynamic_yolo.py
Ultralytics 8.3.202 🚀 Python-3.10.12 torch-2.5.1+cu121 CPU (13th Gen Intel Core i7-13620H)
Model summary (fused): 72 layers, 3,006,038 parameters, 0 gradients, 8.1 GFLOPs

PyTorch: starting from '/mnt/c/Users/Aditri/Desktop/Drone/runs/detect/train4/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 6, 8400) (6.0 MB)
free(): double free detected in tcache 2
Aborted (core dumped)


#WHY ERROR CAME-
from ultralytics import YOLO

model = YOLO("/mnt/c/Users/Aditri/Desktop/Drone/runs/detect/train4/weights/best.pt")
model.export(format="onnx", imgsz=640, opset=12, dynamic=True, simplify=True)
print("Exported best32yolo.onnx")

I ran this code but the problem was due to this line-model.export(format="onnx", imgsz=640, opset=12, dynamic=True, simplify=True)
here.....simplify=True lead to errors
why?
->see my error carefully you will see this line-free(): double free detected in tcache 2
Aborted (core dumped)


THIS MEANS :
it’s a memory management bug in the export script or PyTorch/ONNX stack on my system!

so I RAN SIMPLIFY=FALSE :
why??
->This skips ONNX simplification (often where double free happens)


#ERROR SOLVED
(drone-env) aditrisingh@LAPTOP-6T8704AT:/mnt/c/Users/Aditri/Desktop/Drone$ python3 export_dynamic_yolo.py
Ultralytics 8.3.202 🚀 Python-3.10.12 torch-2.5.1+cu121 CPU (13th Gen Intel Core i7-13620H)
Model summary (fused): 72 layers, 3,006,038 parameters, 0 gradients, 8.1 GFLOPs

PyTorch: starting from '/mnt/c/Users/Aditri/Desktop/Drone/runs/detect/train4/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 6, 8400) (6.0 MB)

ONNX: starting export with onnx 1.19.0 opset 12...
ONNX: export success ✅ 8.1s, saved as '/mnt/c/Users/Aditri/Desktop/Drone/runs/detect/train4/weights/best.onnx' (11.5 MB)

Export complete (10.7s)
Results saved to /mnt/c/Users/Aditri/Desktop/Drone/runs/detect/train4/weights
Predict:         yolo predict task=detect model=/mnt/c/Users/Aditri/Desktop/Drone/runs/detect/train4/weights/best.onnx imgsz=640  
Validate:        yolo val task=detect model=/mnt/c/Users/Aditri/Desktop/Drone/runs/detect/train4/weights/best.onnx imgsz=640 data=uav-multicam/data/det/data.yaml  
Visualize:       https://netron.app
Exported best32yolo.onnx


##this is WHAT ALLL I went through just to-Download and export osnet to onnx


## 1) Initial Setup

Operating System: Windows 11

Environment: Python 3.10 virtual environment (venv)

Packages installed: torch, torchvision, onnx, onnxruntime, torchreid (0.2.5), ultralytics, opencv-python, numpy, tqdm, matplotlib

## 2) Downloading the Pretrained Model

Goal: Get osnet_x0_25_msmt17.pt (trained on MSMT17 dataset, 4101 classes)

Issue: Direct GitHub download failed due to connection aborts.

Solution:

Cloned the Hugging Face YOLOv5 Tracking repo:

git clone https://huggingface.co/spaces/xfys/yolov5_tracking
cd yolov5_tracking
git lfs pull


Verified presence of osnet_x0_25_msmt17.pt in weights/ folder

## 3)Initial Attempt to Export

Used Torchreid import:

from torchreid.models import build_model
from torchreid.utils import load_pretrained_weights


Errors encountered:

ModuleNotFoundError: No module named 'torchreid.utils' → removed in newer Torchreid versions.

Solution: manually load checkpoint and state_dict.


## 4)Handling State Dictionary

Loaded checkpoint manually using:

checkpoint = torch.load("weights/osnet_x0_25_msmt17.pt", map_location="cpu")
state_dict = checkpoint.get('state_dict', checkpoint)


Removed any module. prefix from keys for compatibility

## 5)Classifier Size Mismatch

Initial model num_classes=1000 caused:

size mismatch for classifier.weight: copying a param with shape [4101, 512]


Solution:

Set num_classes=4101 to match checkpoint.

Removed classifier layer entirely using torch.nn.Identity() → only embeddings needed for Re-ID

## 6)ONNX Export

Used dummy input: (1, 3, 256, 128)

Exported using torch.onnx.export with:

dynamic_axes for batch flexibility

Input name: "input"

Output name: "feat"

Final file: weights/osnet_x0_25.onnx ✅








##JOUNREY OF CONVERTING MY OSNET ONNX TO INT8 ENGINE!!
'I am so confused rn about shapes so i chceked the shape of my osnet AGAIN! to be sure before I convert it,you can do that using this siple code snippet :)


import onnx
model = onnx.load("/mnt/c/Users/Aditri/Desktop/Drone/osnet.onnx")  # here use your onnx model's path
print("Input name/shape:", model.graph.input[0].name, [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim])
print("Output name/shape:", model.graph.output[0].name, [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim])

so as expected i got-
Input name/shape: input [0, 3, 256, 128]
Output name/shape: feat [0, 512]



I thought about this quetsion  a lot today -Why is calibration data critical for INT8? What could go wrong with a bad dataset (ex-only daytime images) How does this prep mimic my drone’s real-world tracking?

then i searched and got this answer -
-WHY CALIBRATION DATA IS CRITICAL FOR INT 8?
1)In FP32 numbrs can ber VERY large ,like wide range
2)INT8 has a tiny range :(
3)To fit fp32 values to int8TensorRT needs to learn a scaling factor


#####⚠️ What could go wrong with a bad dataset?

-Imagine you gave only daytime images for calibration:
The model sees only bright images 'only' like..hight contrast pixels during calibration

->IT SETS SCALES EXPECTING ALL IMAGES TO BE BRIGHT

LATER WHEN YOU USE NIGHTTIME IMAGES (DARK,LOW CONTRAST),THE PIXEL VALUES DONT FIT THE LEARNED SCALLE
1)MODEL accuracy might DROPPP
2)IMportant details are lost

This is why people say:

“Calibration is 80% of quantization success.”


##🏴‍☠️⛵⚓🌊 Built osnet.trt(osnet int 8 engine) with 520 images processed##

->(drone-env) aditrisingh@LAPTOP-6T8704AT:/mnt/c/Users/Aditri/Desktop/Drone$ ./trt_int8_osnet
onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
Set dynamic range for feat: [-5,5]
Set dynamic range for (Unnamed Layer* 388) [Constant]_output: [-1,1]
Set dynamic range for (Unnamed Layer* 391) [Constant]_output: [-1,1]
Set dynamic range for (Unnamed Layer* 394) [Constant]_output: [-1,1]
Set dynamic range for (Unnamed Layer* 396) [Constant]_output: [-1,1]
Set dynamic range for (Unnamed Layer* 400) [Constant]_output: [-1,1]
Set dynamic range for (Unnamed Layer* 402) [Constant]_output: [-1,1]
Calibration Profile is not defined. Calibrating with Profile 0
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0001.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0003.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0004.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0005.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0006.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0007.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0008.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0009.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0010.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0011.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0012.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0013.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0014.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0015.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0016.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0020.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0022.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0024.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0025.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0026.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0028.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0029.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0030.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0032.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0033.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0034.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0035.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0037.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0039.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0040.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0042.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0044.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0046.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0048.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0049.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0050.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0052.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0054.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0056.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0058.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0059.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0061.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0063.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0064.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0065.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0068.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0069.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0070.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0071.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0072.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0073.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0074.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0075.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0079.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/0080.jpg (256x128)
Processing image /m.....
.......
......
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_133.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_134.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_135.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_137.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_138.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_140.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_141.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_142.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_143.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_144.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_145.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_146.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_147.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_148.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_149.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_150.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_151.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_152.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_153.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_154.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_155.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_156.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_158.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_159.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_161.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_162.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_164.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_165.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_167.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_169.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_170.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_171.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_172.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_173.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_174.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_175.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_176.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_177.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_179.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_180.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_181.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_182.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_183.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_184.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_185.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_186.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_187.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_188.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_189.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_190.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_192.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_193.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_194.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_195.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_196.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_197.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_198.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_200.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_201.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_203.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_204.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_205.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_206.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_207.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_208.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_209.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_211.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_212.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_213.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_214.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_215.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_217.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_218.jpg (256x128)
Processing image /mnt/c/Users/Aditri/Desktop/Drone/osnet_calib_preprocessed/pic_219.jpg (256x128)
Serialized engine saved to osnet.trt
INT8 engine built successfully for OSNet!
(drone-env) aditrisingh@LAPTOP-6T8704AT:/mnt/c/Users/Aditri/Desktop/Drone$ 
TensorRT requires an optimization profile to define the range of batch sizes (e.g., min=1, opt=32, max=64) for dynamic inputs. Without it, the engine build fails







#:( MY HUGE MISTAKE
I just realised I exported my osnet to onnx with int64 weights and nonw thats what its causing A LITTLEEE precision loss :(
    Look first I tetsed my int8 engine for batch 32 ,it failed but passed batch 1 so I had to again create it for batch 32 I did -


    [09/20/2025-17:17:54] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[09/20/2025-17:17:54] [I] Explanations of the performance metrics are printed in the verbose logs.
[09/20/2025-17:17:54] [V] 
[09/20/2025-17:17:54] [V] === Explanations of the performance metrics ===
[09/20/2025-17:17:54] [V] Total Host Walltime: the host walltime from when the first query (after warmups) is enqueued to when the last query is completed.
[09/20/2025-17:17:54] [V] GPU Compute Time: the GPU latency to execute the kernels for a query.
[09/20/2025-17:17:54] [V] Total GPU Compute Time: the summation of the GPU Compute Time of all the queries. If this is significantly shorter than Total Host Walltime, the GPU may be under-utilized because of host-side overheads or data transfers.
[09/20/2025-17:17:54] [V] Throughput: the observed throughput computed by dividing the number of queries by the Total Host Walltime. If this is significantly lower than the reciprocal of GPU Compute Time, the GPU may be under-utilized because of host-side overheads or data transfers.
[09/20/2025-17:17:54] [V] Enqueue Time: the host latency to enqueue a query. If this is longer than GPU Compute Time, the GPU may be under-utilized.
[09/20/2025-17:17:54] [V] H2D Latency: the latency for host-to-device data transfers for input tensors of a single query.
[09/20/2025-17:17:54] [V] D2H Latency: the latency for device-to-host data transfers for output tensors of a single query.
[09/20/2025-17:17:54] [V] Latency: the summation of H2D Latency, GPU Compute Time, and D2H Latency. This is the latency to infer a single query.
[09/20/2025-17:17:54] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8601] # trtexec --loadEngine=/mnt/c/Users/Aditri/Desktop/Drone/osnet.trt --int8 --calib=/mnt/c/Users/Aditri/Desktop/Drone/int8_osnet.cache --useCudaGraph --useSpinWait --verbose --shapes=input:32x3x256x128
(drone-env) aditrisingh@LAPTOP-6T8704AT:/mnt/c/Users/Aditri/Desktop/Drone$




#BUT UNFORTUNATELY :(
->onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32

Cause: The osnet_x0_25.onnx model uses INT64 weights, but TensorRT 8.6.1 only supports INT32/FP32/FP16 for weights. TensorRT automatically casts to INT32, which is safe for OSNet (minimal accuracy impact)




#SO FINALLY.. I EXPORTED OSNET  WITH INT32

(drone-env) aditrisingh@LAPTOP-6T8704AT:/mnt/c/Users/Aditri/Desktop/Drone$ python3 re_export_osnet.py
/mnt/c/Users/Aditri/Desktop/Drone/drone-env/lib/python3.10/site-packages/torchreid/reid/metrics/rank.py:11: UserWarning: Cython evaluation (very fast so highly recommended) is unavailable, now use python evaluation.
  warnings.warn(
Downloading...
From: https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs
To: /home/aditrisingh/.cache/torch/checkpoints/osnet_x0_25_imagenet.pth
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.97M/2.97M [00:13<00:00, 213kB/s]
/mnt/c/Users/Aditri/Desktop/Drone/drone-env/lib/python3.10/site-packages/torchreid/reid/models/osnet.py:482: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state_dict = torch.load(cached_file)
Successfully loaded imagenet pretrained weights from "/home/aditrisingh/.cache/torch/checkpoints/osnet_x0_25_imagenet.pth"
Exported ONNX with INT32 weights
(drone-env) aditrisingh@LAPTOP-6T8704AT:/mnt/c/Users/Aditri/Desktop/Drone$ 


##😢😢😢😢

After exporting my new int32 osnet onnx to int 8
this is waht i got-
(drone-env) aditrisingh@LAPTOP-6T8704AT:/mnt/c/Users/Aditri/Desktop/Drone$ ./trt_int8_osnet
onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
Loaded 520 calibration images
Read calibration cache from /mnt/c/Users/Aditri/Desktop/Drone/int8_osnet.cache
Read calibration cache from /mnt/c/Users/Aditri/Desktop/Drone/int8_osnet.cache
Missing scale and zero-point for tensor output, expect fall back to non-int8 implementation for any layer consuming or producing given tensor
Engine saved to /mnt/c/Users/Aditri/Desktop/Drone/osnet.trt




##CONCLUSION

->YOLO cant be fully converted into FP16 AND INT8 
->IT WILL ALWAYS RELY ON FP32 FOR SOME LAYERS :)