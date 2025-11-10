# 🧩 Week 2 – Implementation Phase Summary
### Theme : Sustainability  |  Technology : CNN (Convolutional Neural Network)  |  Domain : Image Classification  

---

## ⚙️ Implementation Overview
During **Week 2**, the **Smart Waste Classification System** using **Convolutional Neural Networks (CNN)** was implemented and trained on **Google Colab** (GPU runtime).  
The model used the **Garbage Classification Dataset** from Kaggle to classify images of plastic, paper, metal, glass, and organic waste.  

---

## 🧩 Implementation Steps

### 1️⃣ Dataset Integration
- Imported the **Garbage Classification Dataset** using the **Kaggle API**.  
- Verified folder structure (`/train` and `/val`) and class balance.  

**Dataset Source:** [Kaggle – Garbage Classification v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

---

### 2️⃣ Data Preprocessing
- Used `ImageDataGenerator` for scaling (1/255), rotation, zoom, and horizontal flip.  
- Resized all images to **128 × 128 pixels**.  

---

### 3️⃣ Model Construction
Created a CNN using TensorFlow / Keras with:
- `Conv2D` and `MaxPooling2D` layers for feature extraction.  
- `Flatten` and `Dense` layers for classification.  
- `Dropout` layer to reduce overfitting.  

**Code Snippet (Structure):**
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
