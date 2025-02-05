# Speech Recognition Project

This project is a speech recognition system that converts spoken language into written text. It uses MFCC features and a deep learning model for transcription.

## Directory Structure

<pre>
speech-recognition-project/
│
├── data/              # Dataset storage
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks for analysis
├── scripts/           # Python scripts for training and inference
├── tests/            # Unit tests
├── requirements.txt  # Project dependencies
├── README.md         # Project documentation
└── .gitignore       # Git ignore rules
</pre>

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/speech-recognition-project.git
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the preprocessing script:**
   ```bash
   python scripts/preprocess.py
   ```

4. **Train the model:**
   ```bash
   python scripts/train.py
   ```

5. **Transcribe audio:**
   ```bash
   python scripts/inference.py
   ```
