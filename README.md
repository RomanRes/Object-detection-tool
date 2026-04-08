
# Object Detection Tool based on YOLOv3
### Custom Neural Network Architecture in TensorFlow/Keras & Interactive UI

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-yellow.svg)
![Keras](https://img.shields.io/badge/Manual--Implementation-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## About The Project

This project is an object detection application featuring a manual reconstruction of the YOLOv3 logic. It is specifically optimized to run in a lightweight environment using CPU-only processing to ensure maximum accessibility and efficiency.


<div align="center">
  <img src="https://github.com/RomanRes/Object-detection-tool/blob/main/img/YOLOv3readme.gif" alt="YOLOv3 Detection Tool Demo">
</div>

<br />


## Usage

The app allows you to change the maximum value of the IOU (Intersection over Union) between boxes of the same class, as well as the value of the probability of finding an object in a box (Confidence Threshold), to eliminate unnecessary or redundant detections.

## The Engineering Achievement

While many developers use high-level libraries, I have manually implemented the YOLOv3 architecture and the necessary post-processing algorithms to ensure a deep understanding of the model's behavior.

### What I manually implemented:

*   **Full YOLOv3 Architecture:** I coded the entire 106-layer Darknet-53 backbone and detection heads using the TensorFlow/Keras Functional API. Every Convolutional layer, Residual block, and Skip connection is explicitly defined.
*   **Detection Algorithms:** Custom implementation of Bounding Box Decoding (Sigmoid activations, anchor box scaling, and coordinate normalization) and Non-Maximum Suppression (NMS).
*   **Infrastructure Optimization:** To maintain the smallest possible footprint, the application is built using a Multi-Stage Dockerfile and utilizes TensorFlow 2.15 (CPU-only).

---

## Deployment & Optimization

The Docker image was engineered with a focus on size efficiency and deployment speed:

1.  **Multi-Stage Build:** The build process is divided into two stages. The first stage handles the installation of dependencies and the 240MB weights download, while the second stage only contains the necessary runtime environment and the code.
2.  **CPU Efficiency:** By using the CPU-specific version of TensorFlow 2.15, the image size is significantly reduced compared to GPU-enabled versions, making it easier to share and deploy.
3.  **Docker Hub Support:** To save build time (which can be significant during the initial weights download), a pre-built image will be available for instant execution.

---

## Installation

### Option 1: Fast Start (Pre-built Image)
1. **To save time and avoid the local build process, you can pull and run the pre-built image direktly from Docker Hub:**


   ```bash
   docker run -p 8050:8050 gravitsapapa/object-detection-tool-yolo-app:latest
   ```

### Option 2: Docker Compose (Build from Source)

1. **Clone and Enter:**
   ```bash
   git clone https://github.com/RomanRes/Docker-YOLOv3-Vision-Dashboard.git
   cd Docker-YOLOv3-Vision-Dashboard

   ```

2. **Run:**
   ```bash
   docker-compose up --build
   ```

3. **Explore:**
   Visit http://localhost:8050

---

## Project Components

*   **Architecture:** Manual Keras implementation of Darknet-53.
*   **Algorithms:** Custom NMS and coordinate transformation logic.
*   **Weights:** Automatically integrated YOLOv3 pretrained weights via wget.
*   **Infrastructure:** Optimized Python 3.11-slim Multi-Stage Docker environment.

---

## Why this project stands out

Most portfolios rely on simple pre-built model loaders. This project demonstrates the ability to:

1.  **Translate Research into Code:** Turning the original YOLOv3 paper into a functional Keras architecture.
2.  **Master Complex UI/UX:** Creating an interactive tool for AI parameter testing using Dash and Plotly.
3.  **Optimize Infrastructure:** Managing image size and deployment efficiency using modern DevOps practices like Multi-Stage builds and CPU-only optimization.

---

## Credits and Transparency

*   **Model Architecture:** Based on the YOLOv3 paper by Joseph Redmon.
*   **Weight Handling:** The WeightReader utility is used to parse original binary weights into the Keras layers.
*   **Full Integration and Dashboard:** Developed and engineered by RomanRes.

---

## Contact

**Roman** - [GitHub Profile](https://github.com/RomanRes)
