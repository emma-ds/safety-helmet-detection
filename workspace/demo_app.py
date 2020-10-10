import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


PATH_TO_LABELS = "annotations/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
MODEL_PATH = "my_ssd_mobilenet"
PATH_TO_MODEL_DIR = "exported-models/"
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + MODEL_PATH + "/saved_model"
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

# min_score_threshold = 0.4

# IMAGES_IDX = [8, 64, 153, 390]
# IMAGE_PATHS = ["data/images/hard_hat_workers" + str(i) + ".jpg" for i in IMAGES_IDX]

st.title("Safety Helmet Detection Project")
st.markdown("### Emma Wu, Disha An")

st.markdown("Please upload a raw image")
uploaded_file = st.file_uploader("Upload jpg file", type = ['jpg'])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.markdown("** Original Raw Image: **")
    st.image(uploaded_image)

    ### Preprocess the raw image
    with st.spinner("Pre-processing the image ..."):
        image_np = np.array(uploaded_image)
        # image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]
        st.success("Pre-processing has been done")


    # Make the prediction
    with st.spinner("Making the prediction..."):
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        image_np_with_detections = image_np.copy()
        
        st.markdown("**Detection Threshold: **")
        chosen = st.slider("Please select minimum threshold: ", value=0.5, min_value=0.1, max_value=0.9, step=0.1)
        # chosen = st.radio('Please select minimum threshold: ', ("0.2", "0.4", "0.6", "0.8"))
        st.write(f"The minimum threshold you chose is {chosen}.")
        # chosen = float(chosen)
        min_score_threshold = chosen


        viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,detections['detection_boxes'],detections['detection_classes'],detections['detection_scores'],category_index,use_normalized_coordinates=True,max_boxes_to_draw=200,min_score_thresh=min_score_threshold,agnostic_mode=False)
        plt.figure()
        plt.imshow(image_np_with_detections)
        st.pyplot()

        # print detected classes
        classes, scores = detections['detection_classes'], detections['detection_scores']
        objects = []
        for index, value in enumerate(classes):
            object_dict = {}
            if scores[index] > min_score_threshold:
                object_dict[(category_index.get(value)).get('name').encode('utf8')] = scores[index]
                objects.append(object_dict)

        # raise alerts
        cls = []
        for i in range(len(objects)):
            cls.append(list(objects[i].keys())[0])
            heads_count = sum(i==b'head' for i in cls)
        if heads_count > 0:
            st.markdown("### Alert!! Detected " + str(heads_count) + " people not wearing helmets")
        else:
            st.markdown("### Pass helmet test")