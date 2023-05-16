# Object detection and Streamlit

This repository is a internal project to detect tubes using the YOLOv8 object detection algorithm and streamlit, a Python web application framework for building interactive web applications. 


## Requirements

- Python 3.6+
- YOLOv8 Model
- Ultralytics
- Streamlit

```bash
pip install ultralytics streamlit
```

## Installation
- Start Python environment:
  - `python3 -m venv [Virtual Environment Name]`
  - [Windows]
  - `.\[Virtual Environment Folder Name]\Scripts\activate`
  - [Unix]
  - `source [Virtual Environment Name]/bin/activate`
- Clone the repository: `git clone https://github.com/mqxi/tubedetection_streamlit.git `
- Change to the repository directory: `cd tubedetection_streamlit`
- Install the requirements: `pip install -r requirements.txt`
- Use the tube-trained yolov8 model and save it to the `weights` directory in the same project.

## Usage

- Run the app with the following command: `streamlit run app.py`
- The app should open in a new browser window.

### Debugging
-  If the command prompt loads more than a few seconds to start up: try using an admin cmd prompt to run the app.

### ML Model Config

- Select model confidence
- Use the slider to adjust the confidence threshold (0.25 - 1.0) for the model.
Once the model config is done, select a source.

### Detection on main-webcam

- Select the radio button Webcam.
- Click "Detect Tubes" to run the tube detection algorithm on the live webcam connected to the system.
- The webcam will now be streamed on the page with the colored object detected.
- To end the stream click `Stop` on the top left of the page.


### Detection in Multicamera Mode

- Select the radio button Line Clearance Mode
- Click the "Detect Tubes" Button to run the tube detection algorithm on the live cams connected to the system with the selected confidence threshold.
- Side Info: The cams start right after the click on `Detect Tubes`, 
- The resulting images for each webcam with object detected will be displayed on the page.
- If an image contains a tube, the tube will be colored red and an additional warning sign below the image will be shown.


### Detection on RTSP

- Select the RTSP stream button
- Enter the rtsp url inside the textbox and hit `Detect Tubes` button.


## Acknowledgements

This app is based on the YOLOv8(<https://github.com/ultralytics/ultralytics>) object detection algorithm. The app uses the Streamlit(<https://github.com/streamlit/streamlit>) library for the user interface.
