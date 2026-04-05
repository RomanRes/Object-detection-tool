


<!-- PROJECT LOGO -->
<br />
<div align="center">
  
  <h3 align="center">Object Detection Using YOLO v3 </h3>

</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
        <li><a href="#built-with">Built With</a></li>
    </li>
        <li><a href="#installation">Installation</a></li>


  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<div align="center">
  <img src="https://github.com/RomanRes/Object-detection-tool/blob/main/img/YOLOv3readme.gif" >
</div>

An object detection tools. Based on YOLOv3 with pretrained weights on COCO data set. Can detect 80 classes.





### Built With


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Bootstrap](https://img.shields.io/badge/bootstrap-%23563D7C.svg?style=for-the-badge&logo=bootstrap&logoColor=white)



### Installation

1. Clone the repo
```sh
git clone https://github.com/RomanRes/Object-detection-tool-based-on-Dash-and-YOLOv3.git
```
2. Create a fresh venv (with `conda` or `virtualenv`) and activate it.

3. Install the requirements:
```
pip install -r requirements.txt
```
4. Load weights and put in main folder

   https://pjreddie.com/media/files/yolov3.weights

5. Start the app:

```
python app.py
```
6. Open in browser `http://127.0.0.1:8050/`



<!-- USAGE EXAMPLES -->
## Usage

The app allows you to change the maximum value of the IOU between boxes of the same class, as well as the value of the probability of finding an object in a box, to eliminate unnecessary boxes.


# 🚀 YOLOv3 Manual Implementation & Detection Dashboard
### Custom Neural Network Architecture in TensorFlow/Keras & Interactive UI

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-yellow.svg)
![Keras](https://img.shields.io/badge/Manual--Implementation-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## 💎 The Engineering Achievement
This project is a deep-dive into Computer Vision. While many developers use high-level libraries, I have **manually reconstructed the YOLOv3 logic** to understand and demonstrate the underlying mechanics of modern object detection.

### 🧠 What I manually implemented:
*   **Full YOLOv3 Architecture:** I coded the entire 106-layer Darknet-53 backbone and detection heads using the **TensorFlow/Keras Functional API**. Every Convolutional layer, Residual block, and Skip connection is explicitly defined.
*   **Detection Algorithms (Post-Processing):** I implemented the mathematical logic to transform raw network tensors into meaningful data:
    *   **Coordinate Decoding:** Anchor box scaling and Sigmoid activations.
    *   **Custom NMS (Non-Maximum Suppression):** I wrote the algorithm to filter overlapping detections and select the highest-confidence boxes.
*   **Dash & Plotly UI:** Developed a professional, responsive dashboard from scratch to allow real-time parameter tuning (Confidence & IoU thresholds).
*   **Deployment Infrastructure:** Engineered a multi-stage Docker environment for seamless "one-click" execution.

---

## 🛠 Project Components
*   **Architecture:** Manual Keras implementation of Darknet-53.
*   **Algorithms:** Custom NMS and coordinate transformation logic.
*   **Data Handling:** Optimized pipeline for image resizing, scaling, and batching.
*   **Weight Management:** Integration of a standard `WeightReader` utility to bridge original Darknet `.weights` files with the Keras model.

---

## 📸 App Preview
*(Add your screenshot here)*
`![Detection Results](img/demo_screenshot.png)`

---

## 🚀 Quick Start (Docker Compose)

Experience the full implementation instantly. Docker handles all dependencies and the 240MB weights download automatically.

1.  **Clone & Enter:**
    ```bash
    git clone https://github.com/RomanRes/Object-detection-tool.git
    cd Object-detection-tool
    ```

2.  **Run:**
    ```bash
    docker-compose up --build
    ```

3.  **Explore:**
    Visit [http://localhost:8050](http://localhost:8050)

---

## 📖 Why this project stands out
Most portfolios rely on `model.load()`. This project proves my ability to:
1.  **Translate Research into Code:** Turning the YOLOv3 paper into a functional Keras architecture.
2.  **Master Complex UI/UX:** Creating an interactive tool for AI parameter testing.
3.  **Ensure Portability:** Using Docker to guarantee the code runs everywhere without "dependency hell".

---

## ⚖️ Credits & Transparency
*   **Model Architecture:** Based on the YOLOv3 paper by Joseph Redmon.
*   **Weight Handling:** The `WeightReader` utility is used to parse original binary weights into the Keras layers.
*   **Full Integration & Dashboard:** Developed and engineered by **RomanRes**.

---

## 📧 Contact
**Roman** - [GitHub Profile](https://github.com/RomanRes)


<br />
<div align="center">
  <h3 align="center">Object Detection Tool based on YOLOv3</h3>
  <p align="center">Manual Implementation in TensorFlow and Keras</p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#technical-implementation">Technical Implementation</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#credits">Credits</a></li>
  </ol>
</details>

## About The Project
<div align="center">
  <img src="https://github.com/RomanRes/Object-detection-tool/blob/main/img/YOLOv3readme.gif" alt="Demo">
</div>
<br />
This tool features a manual reconstruction of the YOLOv3 architecture and algorithms. It uses pretrained COCO weights and an interactive Dash dashboard for real-time parameter tuning.

## Technical Implementation
Unlike projects using high-level wrappers, this project features:
* **Manual Architecture:** The 106-layer Darknet-53 backbone was built using the Keras Functional API.
* **Algorithm Engineering:** Custom implementation of Bounding Box Decoding (Sigmoid, Anchors) and Non-Maximum Suppression (NMS).
* **Data Pipeline:** End-to-end processing from Base64 web-upload to normalized 4D tensors.

## Built With
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge) ![Docker](https://img.shields.io/badge/docker-0db7ed?style=for-the-badge)

## Installation
### Option 1: Docker (Recommended)
1. `git clone https://github.com/RomanRes/Object-detection-tool.git`
2. `docker-compose up --build`
3. Access `http://localhost:8050`

## Credits
* **Theory:** Joseph Redmon (YOLOv3 Paper).
* **Weight Utility:** `WeightReader` for binary parsing.
* **Engineering:** Manual implementation and Dash UI by **RomanRes**.

