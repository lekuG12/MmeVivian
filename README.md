# Human Activity Recognition (HAR) Project - Setup & Usage Procedure

This guide explains how to set up, train, and run the HAR system on a new computer from scratch.

---

## 1. **System Requirements**
- Windows 10/11 (recommended) or Linux
- Python 3.8+ (Anaconda or venv recommended)
- At least 8GB RAM (16GB+ recommended for training)
- GPU (CUDA) optional but recommended for faster training

---

## 2. **Clone or Download the Project**
- Download or clone the project folder to your computer.

---

## 3. **Install Python & Create Virtual Environment**

### **Using venv (recommended):**
```bash
python -m venv myenv
# Activate (Windows)
myenv\Scripts\activate
# Activate (Linux/Mac)
source myenv/bin/activate
```

---

## 4. **Install Dependencies**

### **Install from requirements.txt:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **Extra packages for training:**
```bash
pip install tqdm matplotlib seaborn scikit-learn
```

---

## 5. **Prepare the Dataset**
- Place the HMDB51 dataset (or your own) in the `Datasets/` folder.
- The structure should be:
  ```
  Datasets/
    action1/
      video1.avi
      video2.avi
      ...
    action2/
      ...
    ...
  ```
- Each subfolder is a class (action), and each contains `.avi` videos.

---

## 6. **Training the Model**
- Open `improved_har_training.ipynb` in Jupyter or Colab.
- Update the `dataset_path` variable if needed.
- Run all cells in order.
- The notebook will:
  - Train the model with data augmentation, validation, and early stopping
  - Save the best model to `model/improved_har_model_best.pth`
  - Save class names to `model/class_names.json`
  - Save training curves and confusion matrix images

---

## 7. **Running Inference (Detection)**

### **Option 1: PyTorch Model**
- Use `Video/detection_onnx_fallback.py` (works with both PyTorch and ONNX models)
- By default, it will use the improved PyTorch model if ONNX is not available
- To run:
```bash
python Video/detection_onnx_fallback.py
```
- The script will:
  - Load the trained model and class names
  - Process a test video (default: `test_videos/walk.mp4`)
  - Display predictions and send WhatsApp messages (if configured)

### **Option 2: ONNX Model (Optional)**
- If you export your model to ONNX, install ONNX Runtime:
```bash
pip install onnxruntime
```
- Place your ONNX model in `model/HAR.onnx`
- The fallback script will use ONNX if available

---

## 8. **WhatsApp Messaging Setup**
- The detection script uses `pywhatkit` to send WhatsApp messages
- You must be logged into WhatsApp Web on your default browser
- Update the phone number in the script if needed:
  ```python
  PHONE_NUMBER = "+237XXXXXXXXX"
  ```
- The script will only send a message if a new action is detected and after a cooldown period

---

## 9. **Customizing the Project**
- **Change test video:** Edit the `VIDEO_SOURCE` variable in the detection script
- **Change actions:** Edit `actions.txt` and retrain the model
- **Change model architecture:** Edit the model class in the training notebook

---

## 10. **Troubleshooting**
- **DLL errors with ONNX Runtime:** Use the fallback script or reinstall ONNX Runtime
- **CUDA errors:** Make sure you have the correct CUDA toolkit and drivers
- **Low accuracy:** Train for more epochs, use more data, or improve the model
- **WhatsApp not sending:** Make sure WhatsApp Web is open and you are logged in

---

## 11. **Files & Folders Overview**
- `requirements.txt` — Python dependencies
- `improved_har_training.ipynb` — Main training notebook
- `Video/detection_onnx_fallback.py` — Detection script (PyTorch/ONNX)
- `actions.txt` — List of action classes
- `Datasets/` — Folder for your video dataset
- `test_videos/` — Folder for test videos
- `model/` — Folder for saved models and class names

---

## 12. **Contact & Support**
- For issues, check the README or contact the project maintainer.

---
