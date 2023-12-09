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

   ```bash
   python -m pip install --upgrade pip
   ```

   b. **MacOS/Linux**

   ```bash
   python3 -m venv env
   ```

   ```bash
   source env/bin/activate
   ```

   ```bash
   python3 -m pip install --upgrade pip
   ```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

If the above command fails, try the following:

```bash
pip install tensorflow streamlit Pillow==9.5.0
```

### Running the Application

```bash
streamlit run app.py
```

You can now view the application in your browser at `localhost:8501`.
To utilize the application, you will need to upload an image of a hand gesture (we have provided some test images to use in the `images/asl_alphabet_test/` directory).
Our model will then predict the letter that the hand gesture represents along with the model's confidence in the prediction.
