# Drowsiness Detection

This project detects drowsiness using computer vision and deep learning. It uses Haar Cascade classifiers for face and eye detection with OpenCV, and a trained deep learning model to determine if the user is drowsy. An alarm will sound if drowsiness is detected.

## Project Structure

```bash
drowsiness_detection/
│
├── haar cascade files/ # Haar cascade XML files for face and eye detection
│ └── [*.xml]
│
├── models/ # Pretrained deep learning models
│ └── [model.h5]
│
├── alarm2.wav # Alarm sound for drowsiness detection
├── short-success-sound.wav # Notification sound for testing
│
├── main.py # Main application script
├── test.ipynb # Jupyter notebook for testing and experimenting
├── requirements.txt # Required Python libraries
└── README.md # Project documentation
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/drowsiness_detection.git
cd drowsiness_detection
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

- To start drowsiness detection, simply run:

```bash
python main.py
```

- Make sure your webcam is enabled, and the necessary Haar cascade files and model files are available in the correct folders.

- If drowsiness is detected, an alarm (alarm2.wav) will sound.

- You can also use test.ipynb to experiment with loading models, trying out detection snippets, and adjusting settings.

## Requirements

- python(3.9.\*)

  The project dependencies are listed in requirements.txt, including:

- OpenCV
- TensorFlow / Keras
- NumPy
- Other standard Python libraries

Make sure your Python environment satisfies the versions required.

## Notes

- Haar cascades are used for initial face and eye detection.
- The deep learning model (H5 file) further classifies the eye state (open/closed).
- The project is tested on real-time webcam footage.

## Acknowledgements

- OpenCV for providing the Haar Cascade classifiers.
- TensorFlow/Keras for deep learning frameworks.
- Various online tutorials and datasets that helped in model training.
