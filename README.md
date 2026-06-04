# Flood Detection with Interface and XAI

This is a flood detection image classification project. The goal is to compare multiple deep learning architectures for the same task and provide a clear, interactive way to test them.

## Project Summary

- Task: binary image classification (Flood vs Not Flood).
- Models compared: ResNet-50, Inception V3, ViT B-16, and a custom CNN.
- Result: ResNet-50 achieved the best overall classification performance, while the custom CNN is the fastest at inference time.

## Interface and API

The project includes two connected components:

1) An interactive Streamlit interface for uploading images, selecting a model, and viewing predictions, confidence, and inference time.
2) A FastAPI backend that loads the selected model and returns predictions through a simple REST endpoint.

## Installation

Create a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

You need two terminals: one for the API and one for the Streamlit interface.

Terminal 1 (FastAPI):

```bash
uvicorn api:app --reload --port 8000
```

Terminal 2 (Streamlit UI):

```bash
streamlit run app.py
```

Open the Streamlit URL shown in the terminal to use the interface.
