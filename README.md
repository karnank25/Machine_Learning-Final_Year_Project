# Knee Osteoarthritis Detection Using Deep Learning

## Overview
This project aims to develop an automated system for detecting and classifying **Knee Osteoarthritis (KOA)** from X-ray images using deep learning techniques. The system utilizes advanced models, including **CenterNet** for object detection and **DenseNet-201** for feature extraction, to detect KOA severity levels (Grades I-IV). The implementation of **Knowledge Distillation** optimizes performance by allowing a lightweight **student model** to learn from a larger, complex **teacher model**, ensuring high accuracy with reduced computational cost.

---

## Features
- Automatic KOA classification (Healthy, Grade I-IV)
- Optimized, fast detection using Knowledge Distillation
- Scalable and deployable solution for healthcare environments
- Real-time X-ray image analysis

---

## Technologies Used
- **Programming Language:** Python  
- **Frameworks:** TensorFlow, Keras  
- **Deep Learning Models:** CenterNet, DenseNet-201  
- **Image Processing:** X-ray preprocessing and analysis  

---

## Project Structure
```plaintext
- /data: X-ray image dataset
- /models: Trained teacher and student models
- /notebooks: Model training and testing notebooks
- main.py: Script for running the detection system
- requirements.txt: Required packages
- README.md: Project documentation
```

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/knee-osteo-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd knee-osteo-detection
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Place your X-ray images in the `/data` directory.
2. Run the detection system:
   ```bash
   python main.py
   ```
3. View results, including KOA severity predictions.

---

## Results
- High accuracy detection of KOA severity levels
- Lightweight student model with fast processing
- Successful testing on unseen data

---

## Applications
- Early diagnosis of Knee Osteoarthritis
- Clinical decision support for radiologists
- Remote healthcare diagnosis (Telemedicine)
- Patient monitoring and follow-ups

---

## Contributing
Feel free to submit issues or pull requests for improvements.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.
