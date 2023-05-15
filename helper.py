# External Packages
from ultralytics import YOLO
import streamlit as st
import cv2

# Local import
import settings


@st.cache_data
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url")
    #is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Tubes'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image
                                             )
                    # is_display_tracker,
                    # tracker
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    #is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Tubes'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image
                                             )
                    # is_display_tracker,
                    # tracker
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


@st.cache_data
def get_available_camera_indices():
    available_cameras = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        available_cameras.append(index)
        cap.release()
        index += 1
    return available_cameras

# Define a function to resize the annotated frame to match the original image size


def resize_annotated_frame(annotated_frame, original_image):
    return cv2.resize(annotated_frame, (original_image.shape[1], original_image.shape[0]))


def play_multi_webcam_first(conf, model):
    """
    Plays multiple webcam streams. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    #source_webcams = settings.WEBCAM_PATH_LIST
    source_webcams = get_available_camera_indices()
    vid_caps = [cv2.VideoCapture(src) for src in source_webcams]
    st_frames = [st.empty() for _ in range(len(source_webcams))]

    processed_images = 0
    if st.sidebar.button('Enable Cameras'):

        while processed_images < len(source_webcams):
            # Take a picture with every available camera in the system
            images = []
            for idx, vid_cap in enumerate(vid_caps):
                success, image = vid_cap.read()
                if success:
                    images.append(image)
                else:
                    vid_cap.release()
                    vid_caps[idx] = cv2.VideoCapture(source_webcams[idx])
                    images.append(None)

            # Detect objects in each picture using YOLOv8
            objects_detected = []
            for image in images:
                if image is None:
                    objects_detected.append(None)
                else:
                    objects_detected.append(model.predict(image, conf=conf))

            # Display every camera as compact as possible
            cols = st.columns(len(source_webcams))
            for idx, col in enumerate(cols):
                col.write(f"Camera {idx}")
                if images[idx] is None:
                    col.write("Error loading video")
                else:
                    # Run YOLOv8 inference on the image
                    results = model(images[idx])

                    # Get the annotated frame and resize it to match the original image size
                    annotated_frame = resize_annotated_frame(
                        results[0].plot(), images[idx])

                    # Display the image
                    col.image(annotated_frame, channels="RGB")

                    if objects_detected[idx] is not None:
                        col.warning("Object detected!", icon="⚠️")
                        # Highlight the camera with the detected object
                    else:
                        col.empty()
            processed_images += 1


def play_multi_webcam2(conf, model):
    """
    Plays multiple webcam streams. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcams = get_available_camera_indices()
    vid_caps = [cv2.VideoCapture(src) for src in source_webcams]
    st_frames = [st.empty() for _ in range(len(source_webcams))]

    processed_images = 0
    if st.sidebar.button('Enable Cameras'):
        while processed_images < len(source_webcams):
            # Take a picture with every available camera in the system
            images = [vid_cap.read()[1] if vid_cap.isOpened()
                      else None for vid_cap in vid_caps]

            # Detect objects in each picture using YOLOv8
            objects_detected = [model.predict(
                image, conf=conf) if image is not None else None for image in images]

            # Display every camera as compact as possible
            cols = st.columns(len(source_webcams))
            for idx, col in enumerate(cols):
                col.write(f"Camera {idx}/{source_webcams[idx]}")
                if images[idx] is None:
                    col.write("Error loading video")
                else:
                    # Run YOLOv8 inference on the image
                    results = model(images[idx])

                    # Get the annotated frame and resize it to match the original image size
                    annotated_frame = resize_annotated_frame(
                        results[0].plot(), images[idx])

                    # Display the combined image
                    col.image(annotated_frame, channels="RGB")

                    if objects_detected[idx] is not None:
                        col.warning("Object detected!", icon="⚠️")
                        # Highlight the camera with the detected object
                        # col.set_background_color("#FF0000")
                    else:
                        col.empty()
            processed_images += 1


def read_images_from_captures(captures):
    images = []
    for vid_cap in captures:
        try:
            success, image = vid_cap.read()
            images.append(image if success else None)
        except:
            vid_cap.release()
            images.append(None)
    return images


def play_multi_webcam_test(conf, model):
    """
    Plays multiple webcam streams. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    #source_webcams = settings.WEBCAM_PATH_LIST
    source_webcams = get_available_camera_indices()
    vid_caps = [cv2.VideoCapture(src) for src in source_webcams]
    st_frames = [st.empty() for _ in range(len(source_webcams))]

    processed_images = 0
    if st.sidebar.button('Enable Cameras'):

        while processed_images < len(source_webcams):
            # Take a picture with every available camera in the system
            images = read_images_from_captures(vid_caps)

            # Detect objects in each picture using YOLOv8
            objects_detected = []
            for image in images:
                if image is None:
                    objects_detected.append(None)
                else:
                    objects_detected.append(model.predict(image, conf=conf))

            # Display every camera as compact as possible
            cols = st.columns(len(source_webcams))
            for idx, col in enumerate(cols):
                col.write(f"Camera {idx}")
                if images[idx] is None:
                    col.write("Error loading video")
                else:
                    # Run YOLOv8 inference on the image
                    results = model(images[idx])

                    # Get the annotated frame and resize it to match the original image size
                    annotated_frame = resize_annotated_frame(
                        results[0].plot(), images[idx])

                    # Display the image
                    col.image(annotated_frame, channels="RGB")

                    if objects_detected[idx] is not None:
                        col.warning("Object detected!", icon="⚠️")
                        # Highlight the camera with the detected object
                    else:
                        col.empty()
            processed_images += 1
