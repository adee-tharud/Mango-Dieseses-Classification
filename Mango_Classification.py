import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model

model = load_model('NewModel.h5')
lab = {0: "Anthracnose", 1: "powdery mildew", 2: "Mango scab", 3: "healthy", 4: "Red rust"}


def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res


def run():
    st.markdown("<h1 style='text-align: center; font-size:100px; font-family:jokerman; color: green;'>MangoFix</h1>", unsafe_allow_html=True)

    image = Image.open('245-2451906_why-hafoos-mango-png-mango-clipart-cartoon-mango-removebg-preview.png')
    st.image(image, width=400)
    st.markdown(
        '''<h4 style='text-align: left; color: #A5EE24;'>* Data is based "1000  Mango Diseases"</h4>''',
        unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; color: black;'><b>Mango fruit is in high demand. So, the timely control of mango plant diseases is necessary to "
                "gain high returns. Automated recognition of mango plant leaf diseases is still a challenge as manual "
                "disease detection is not a feasible choice in this computerized era due to its high cost and the "
                "non-availability of mango experts and the variations in the symptoms. Amongst all the challenges, "
                "the segmentation of diseased parts is a big issue, being the pre-requisite for correct recognition "
                "and identification. For this purpose, a novel segmentation approach is proposed in this study to "
                "segment the diseased part by considering the vein pattern of the leaf</b></p>", unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Mango Leaves", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,width=300, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Classify"):
            result = processed_img(save_image_path)
            st.success("Predicted Mango Leave is: " + result)


run()
