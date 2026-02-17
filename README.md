🛡️ Violence Detection System (CNN-LSTM)
An automated video classification system designed to identify violent activities in real-time using Deep Learning. This project leverages the temporal relationship between video frames to distinguish between "Violence" and "Non-Violence" with high precision.

🧠 Model Architecture
The system utilizes a hybrid architecture that combines spatial and temporal feature learning:

Feature Extraction (Spatial): Uses a pre-trained MobileNetV2 model (frozen ImageNet weights) to extract high-level visual features from individual frames.

Sequence Modeling (Temporal): A Long Short-Term Memory (LSTM) network processes these features sequentially to understand movement and action context over time.

Input Shape: The model processes sequences of 16 frames, each resized to 160x160 pixels.

🚀 Key Features
Memory Efficient: Implements a custom Data Generator to load and process video batches on the fly, avoiding GPU memory crashes.

Standardized Sampling: Automatically extracts exactly 16 representative frames from videos of any length to maintain consistency.

Evaluation Metrics: Includes a comprehensive Confusion Matrix and classification report to monitor False Positives and False Negatives.

🛠️ Tech Stack
Language: Python

Frameworks: TensorFlow / Keras

Computer Vision: OpenCV

Data Handling: NumPy, Pandas
