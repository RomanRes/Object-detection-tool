
# Object Detection Tool based on YOLOv3
### Custom Neural Network Architecture in TensorFlow/Keras & Interactive UI

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-yellow.svg)
![Keras](https://img.shields.io/badge/Manual--Implementation-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## About The Project

<div align="center">
  <img src="https://github.com/RomanRes/Object-detection-tool/blob/main/img/YOLOv3readme.gif" alt="YOLOv3 Detection Tool Demo">
</div>

<br />

This project is an object detection application that features a manual reconstruction of the YOLOv3 logic to demonstrate the underlying mechanics of modern computer vision.

## Usage

The app allows you to change the maximum value of the IOU (Intersection over Union) between boxes of the same class, as well as the value of the probability of finding an object in a box (Confidence Threshold), to eliminate unnecessary or redundant detections.

## The Engineering Achievement

While many developers use high-level libraries, I have manually implemented the YOLOv3 architecture and the necessary post-processing algorithms to ensure a deep understanding of the model's behavior.

### What I manually implemented:

*   **Full YOLOv3 Architecture:** I coded the entire 106-layer Darknet-53 backbone and detection heads using the TensorFlow/Keras Functional API. Every Convolutional layer, Residual block, and Skip connection is explicitly defined.
*   **Detection Algorithms (Post-Processing):** I implemented the mathematical logic to transform raw network tensors into meaningful data:
    *   **Coordinate Decoding:** Anchor box scaling and Sigmoid activations.
    *   **Custom NMS (Non-Maximum Suppression):** I wrote the algorithm to filter overlapping detections and select the highest-confidence boxes.
*   **Dash & Plotly UI:** Developed a professional, responsive dashboard from scratch to allow real-time parameter tuning (Confidence and IoU thresholds).
*   **Deployment Infrastructure:** Engineered a Docker environment for seamless execution on any platform.

---

## Project Components

*   **Architecture:** Manual Keras implementation of Darknet-53.
*   **Algorithms:** Custom NMS and coordinate transformation logic.
*   **Data Handling:** Optimized pipeline for image resizing, scaling, and batching.
*   **Weight Management:** Integration of a standard WeightReader utility to bridge original Darknet .weights files with the Keras model.

---

## Quick Start (Docker Compose)

The application can be started instantly using Docker. The environment handles all dependencies and the 240MB weights download automatically.

1.  **Clone and Enter:**
    ```bash
    git clone https://github.com/RomanRes/Object-detection-tool.git
    cd Object-detection-tool
    ```

2.  **Run:**
    ```bash
    docker-compose up --build
    ```

3.  **Explore:**
    Visit http://localhost:8050

---

## Why this project stands out

Most portfolios rely on simple pre-built model loaders. This project demonstrates the ability to:

1.  **Translate Research into Code:** Turning the original YOLOv3 paper into a functional Keras architecture.
2.  **Master Complex UI/UX:** Creating an interactive tool for AI parameter testing.
3.  **Ensure Portability:** Using Docker to guarantee the code runs across different platforms without dependency conflicts.

---

## Credits and Transparency

*   **Model Architecture:** Based on the YOLOv3 paper by Joseph Redmon.
*   **Weight Handling:** The WeightReader utility is used to parse original binary weights into the Keras layers.
*   **Full Integration and Dashboard:** Developed and engineered by RomanRes.

---

## Contact

**Roman** - [GitHub Profile](https://github.com/RomanRes)