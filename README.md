# ASL-Recognition

This project focuses on recognizing American Sign Language (ASL) gestures.

## Team Members

-  **Alexander Hoang**
-  **Kay Krachenfels**
-  **Nicholas Mueller**
-  **Pavittar Singh**
-  **Siddarth Vinnakota**

## Getting Started

Follow these steps to set up the project environment and run the application.

### Setting Up the Environment

1. **Create a Virtual Environment**

   a. **Windows**

   ```bash
   python -m venv env
   ```

   ```bash
   env\Scripts\activate.bat
   ```

   b. **MacOS/Linux**

   ```bash
   python3 -m venv env
   ```

   ```bash
   source env/bin/activate
   ```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

You can now view the application in your browser at `localhost:8501`.
To utilize the application, you will need to upload an image of a hand gesture.
Our model will then predict the letter that the hand gesture represents along with the model's confidence in the prediction.
