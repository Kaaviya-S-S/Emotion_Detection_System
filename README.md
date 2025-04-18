# 🎭 Real-Time Emotion Detection System
This project presents a real-time emotion detection system using **Convolutional Neural Networks (CNN) enhanced with Uniform Local Binary Patterns (uLBP)** for robust facial feature extraction. The system is designed with data augmentation techniques to improve model generalization, making it highly applicable in human-computer interaction and mental health monitoring domains.  

---

## 📌 Features  
- 🎞️ Real-time facial emotion recognition using webcam input  
- 🧠 CNN architecture optimized for emotion classification  
- 🔄 uLBP integration for texture-based facial feature enhancement  
- 🌐 Data augmentation to prevent overfitting and improve accuracy
- 🛠️ Built using Python, TensorFlow, and OpenCV
   
---

## 🚀 Get Started  
📚 **Dataset: FER-2013 (Facial Expression Recognition):**  
🔗 https://www.kaggle.com/datasets/msambare/fer2013

#### 1. Clone this repository  
``` bash
git clone https://github.com/Kaaviya-S-S/Emotion_Detection_System.git
cd Emotion_Detection_System
```
#### 2. Run data augmentation and preprocessing
``` bash
python PreprocessingData.py
```
#### 3. Split data into train, validation, and test sets 
``` bash
python BuildingCNN.py
```
#### 4. Build the CNN model
``` bash
python splitData.py
```
#### 5.Train the model  
``` bash
python CNNModel.py
```
#### 6. Evaluate and test the model  
``` bash
python TestingModel.py
```
---

## 📄 Paper Reference
📝 The original research paper is included as *paper.pdf* in this repository for reference.  

---

## 📃 License  
⚖️This project is licensed under the MIT License – Feel free to explore, modify, and enhance!  

---


