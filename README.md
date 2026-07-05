\# 🚦 Smart Traffic Monitoring System



\### Real-Time Vehicle Detection, Multi-Object Tracking and Traffic Analytics using CARLA, YOLOv8 and Streamlit



!\[Python](https://img.shields.io/badge/Python-3.10-blue)

!\[YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-success)

!\[CARLA](https://img.shields.io/badge/CARLA-Simulator-orange)

!\[Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

!\[OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-blueviolet)



\---



\## 📌 Project Overview



The \*\*Smart Traffic Monitoring System\*\* is an AI-powered intelligent traffic analysis platform that combines computer vision, autonomous driving simulation, and real-time analytics to monitor road traffic.



The system leverages the \*\*CARLA Simulator\*\* to generate realistic urban traffic scenarios and employs the \*\*YOLOv8\*\* object detection model to identify and classify vehicles in real time. Vehicle trajectories are maintained using \*\*Deep SORT Multi-Object Tracking\*\*, enabling continuous monitoring across video frames.



Traffic statistics generated from the simulation are streamed to an interactive \*\*Streamlit Dashboard\*\*, providing live visualization of traffic density, vehicle counts, and system performance.



The project demonstrates how Artificial Intelligence and Intelligent Transportation Systems (ITS) can be integrated to improve urban traffic monitoring and management while providing a scalable platform for future smart city applications.



\---



\## ✨ Key Features



\- 🚗 Real-time vehicle detection using YOLOv8

\- 🎯 Deep SORT based multi-object tracking

\- 🌆 Traffic simulation using CARLA

\- 📈 Interactive Streamlit dashboard

\- 📡 MQTT-based traffic data publishing

\- 📊 Model performance evaluation

\- ⚡ Lightweight real-time inference

\- 📂 Modular project architecture

\- 🔬 Research-oriented implementation



\---



\## 🎯 Objectives



\- Automate vehicle detection in urban environments.

\- Track vehicles across consecutive frames.

\- Generate real-time traffic analytics.

\- Visualize traffic information through an interactive dashboard.

\- Evaluate object detection performance using standard metrics.

\- Provide a scalable architecture for Intelligent Transportation Systems.



\---



\## 🏗️ High-Level Workflow



```text

CARLA Simulation

       │

       ▼

RGB Camera Stream

       │

       ▼

YOLOv8 Vehicle Detection

       │

       ▼

Deep SORT Tracking

       │

       ▼

Traffic Statistics

       │

       ▼

MQTT Publisher

       │

       ▼

Streamlit Dashboard

``---



\# 🏛️ System Architecture



The project follows a modular pipeline that integrates simulation, computer vision, object tracking, messaging, and visualization into a unified traffic monitoring system.



```text

&#x20;                    ┌──────────────────────────┐

&#x20;                    │     CARLA Simulator      │

&#x20;                    └────────────┬─────────────┘

&#x20;                                 │

&#x20;                                 ▼

&#x20;                    RGB Camera Sensor Stream

&#x20;                                 │

&#x20;                                 ▼

&#x20;                 YOLOv8 Vehicle Detection Model

&#x20;                                 │

&#x20;                                 ▼

&#x20;                Deep SORT Multi-Object Tracking

&#x20;                                 │

&#x20;                ┌────────────────┴────────────────┐

&#x20;                ▼                                 ▼

&#x20;         Vehicle Counting                 Traffic Statistics

&#x20;                │                                 │

&#x20;                └────────────────┬────────────────┘

&#x20;                                 ▼

&#x20;                          MQTT Publisher

&#x20;                                 │

&#x20;                                 ▼

&#x20;                        InfluxDB Time-Series DB

&#x20;                                 │

&#x20;                                 ▼

&#x20;                     Streamlit Analytics Dashboard

```



\---



\# 📁 Repository Structure



```text

Smart-Traffic-Monitoring-System/

│

├── simulation/

│   ├── Vehicle detection scripts

│   ├── CARLA simulation utilities

│   ├── Sensor processing

│   ├── MQTT publisher

│   └── YOLO inference pipeline

│

├── dashboard/

│   ├── Streamlit application

│   ├── MJPEG streaming

│   └── Dashboard dependencies

│

├── model-training/

│   ├── Training results

│   ├── Precision-Recall curves

│   ├── Confusion matrices

│   ├── Labels

│   └── Performance metrics

│

├── screenshots/

│

├── demo/

│

├── docs/

│

└── README.md

```



\---



\# 🛠️ Technology Stack



| Category | Technologies |

|----------|--------------|

| Programming Language | Python |

| Simulation | CARLA Simulator |

| Deep Learning | YOLOv8 (Ultralytics) |

| Computer Vision | OpenCV |

| Object Tracking | Deep SORT |

| Dashboard | Streamlit |

| Messaging | MQTT |

| Database | InfluxDB |

| Data Processing | NumPy, Pandas |

| Visualization | Matplotlib |



\---



\# ⚙️ Installation



\## 1. Clone the repository



```bash

git clone https://github.com/Anupam1522/Smart-Traffic-Monitoring-System.git

cd Smart-Traffic-Monitoring-System

```



\## 2. Install dependencies



```bash

pip install -r simulation/requirements.txt

pip install -r dashboard/requirements.txt

```



\## 3. Launch CARLA Simulator



Start the CARLA server and load the required town before running the simulation.



\## 4. Run the simulation



```bash

python simulation/generate\_traffic.py

```



\## 5. Launch the dashboard



```bash

streamlit run dashboard/app\_carla.py

````

\---



\# 📊 Model Performance



The object detection model was trained and evaluated using the YOLOv8 Nano architecture on a custom traffic dataset. The model demonstrated reliable real-time inference while maintaining a lightweight deployment footprint suitable for simulation-based traffic analytics.



\## Evaluation Metrics



The model was evaluated using standard object detection metrics including:



\- Precision

\- Recall

\- F1-Score

\- Mean Average Precision (mAP)

\- Confusion Matrix

\- Precision-Recall Curve



These metrics were generated after training and are available in the `model-training/` directory.



\---



\# 📈 Training Results



The repository includes the complete training artifacts generated during model development.



| File | Description |

|------|-------------|

| results.png | Overall training performance |

| results.csv | Training statistics |

| confusion\_matrix.png | Detection confusion matrix |

| confusion\_matrix\_normalized.png | Normalized confusion matrix |

| BoxPR\_curve.png | Precision-Recall Curve |

| BoxP\_curve.png | Precision Curve |

| BoxR\_curve.png | Recall Curve |

| BoxF1\_curve.png | F1 Score Curve |

| labels.jpg | Dataset label visualization |

| train\_batch0.jpg | Sample training batch |

| train\_batch1.jpg | Sample training batch |

| train\_batch2.jpg | Sample training batch |



These artifacts provide quantitative evidence of the detector's performance and training convergence.



\---



\# 🚀 Key Contributions



This project demonstrates the integration of multiple technologies into a unified Intelligent Transportation System.



Major contributions include:



\- Real-time vehicle detection using YOLOv8.

\- Multi-object tracking using Deep SORT.

\- CARLA-based urban traffic simulation.

\- MQTT-based communication pipeline.

\- Streamlit dashboard for traffic visualization.

\- Modular project architecture for scalability.

\- Performance evaluation using standard computer vision metrics.

\- End-to-end intelligent traffic monitoring workflow.



\---



\# ⚡ Challenges Addressed



\- Reliable vehicle detection under varying traffic density.

\- Maintaining object identities across consecutive frames.

\- Real-time processing within simulation constraints.

\- Efficient communication between simulation and dashboard.

\- Lightweight deployment suitable for research and experimentation.



\---



\# 🔮 Future Enhancements



The current implementation establishes a strong foundation for future Intelligent Transportation System research.



Possible future improvements include:



\- Reinforcement Learning based adaptive traffic signal control.

\- Multi-camera traffic monitoring.

\- Edge deployment on NVIDIA Jetson devices.

\- Emergency vehicle prioritization.

\- Traffic congestion prediction using deep learning.

\- Integration with real-world CCTV feeds.

\- Cloud deployment for city-scale monitoring.

\- Automatic incident detection.



\---



\# 👥 Contributors



This project was developed collaboratively as part of a team.



\*\*Contribution Statement\*\*



> I was one of the developers of this project and contributed to the design, implementation, integration, testing, documentation, and deployment of the Smart Traffic Monitoring System.



\---



\# 📄 License



This repository is intended for educational and research purposes.



Please contact the contributors before using substantial portions of the project for commercial applications.



\---



\# 🙏 Acknowledgements



Special thanks to:



\- CARLA Simulator

\- Ultralytics YOLOv8

\- Streamlit

\- OpenCV

\- Deep SORT

\- Python Open Source Community



Their excellent tools and libraries made this project possible.



\---



\# 📚 References



1\. CARLA Simulator Documentation

2\. Ultralytics YOLOv8 Documentation

3\. Deep SORT Paper

4\. OpenCV Documentation

5\. Streamlit Documentation



\---



\# ⭐ Repository Status



✅ Simulation Module Completed



✅ Dashboard Module Completed



✅ Model Training Completed



✅ Documentation Completed



⬜ Demo Media (In Progress)



⬜ Screenshots (In Progress)



\---



\## If you found this repository useful, consider giving it a ⭐.

