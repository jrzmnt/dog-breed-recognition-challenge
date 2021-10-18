import numpy as np
import streamlit as st
import numpy as np
import pickle
from model import DogModel
from joblib import load
from PIL import Image
from torchvision import transforms


def feature_extractor(model, image_path):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225))])

    img = data_transforms(img).unsqueeze(0)
    features = model._forward_features(img)

    return features


@st.cache(allow_output_mutation=True)
def load_knn(filename):
    knn = load(filename)

    return knn


@st.cache(allow_output_mutation=True)
def load_cnn(filename):
    model = DogModel.load_from_checkpoint(filename)

    return model


ENROLL_DICT = './../data/enroll_dict.pkl'
MODEL_CKPT_DIR = './../models/resnet-50/'
MODEL_CKPT = 'save-epoch=03-val_loss=0.44.ckpt'
KNN_BREED_PATH = './../models/knn-breed/knn_breed.joblib'
KNN_UNKNOWN_PATH = './../models/knn-unknown/knn_unknown.joblib'
img_path = './../data/dogs/recognition/enroll/n02092002-Scottish_deerhound/n02092002_12394.jpg'

enroll_dict = {0: 'West_Highland_white_terrier',
               1: 'groenendael',
               2: 'Scottish_deerhound',
               3: 'Kerry_blue_terrier',
               4: 'Rhodesian_ridgeback',
               5: 'African_hunting_dog',
               6: 'golden_retriever',
               7: 'Eskimo_dog',
               8: 'Norwich_terrier',
               9: 'Cardigan',
               10: 'English_springer',
               11: 'Samoyed',
               12: 'Great_Pyrenees',
               13: 'Bernese_mountain_dog',
               14: 'redbone',
               15: 'Australian_terrier',
               16: 'Mexican_hairless',
               17: 'Rottweiler',
               18: 'coated_retriever',
               19: 'miniature_schnauzer'}


with st.spinner('Model is being loaded..'):
    knn_breed = load_knn(KNN_BREED_PATH)
    knn_unknown = load_knn(KNN_UNKNOWN_PATH)
    model = load_cnn(MODEL_CKPT_DIR+MODEL_CKPT)

st.write("""
         # Dog Breed Recognition
         """
         )


# drag and drop para a aplicação
file = st.file_uploader(
    "Please, upload your image", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)


if file is None:
    st.text("Please, upload an image file.")

else:
    features = feature_extractor(model, file)
    image = Image.open(file)

    st.image(image, use_column_width=False, width=224)

    unknown_predict = knn_unknown.predict(features)

    # 0 == Enroll  # 1 == Unknown
    if not unknown_predict:
        breed_predict = knn_breed.predict(features)
        breed_probs = knn_breed.predict_proba(features)

        st.write(
            f"Your image seems like a **{enroll_dict[breed_predict[0]]}**!")

        # st.write(
        #    f"Also known as **class {breed_predict[0]}**!")

        # top-n argsort - crétitos para: https://stackoverflow.com/a/27433395
        arr = np.array(breed_probs)
        idx = (-arr).argsort()[:3]

        top1 = arr[0][idx[0][0]] * 100.
        top2 = arr[0][idx[0][1]] * 100.
        top3 = arr[0][idx[0][2]] * 100.

        col1, col2, col3 = st.columns(3)
        col1.metric(f"Top 1 ({enroll_dict[idx[0][0]]})", f"{top1:.2f}%")
        col2.metric(f"Top 2 ({enroll_dict[idx[0][1]]})", f"{top2:.2f}%")
        col3.metric(f"Top 3 ({enroll_dict[idx[0][2]]})", f"{top3:.2f}%")

    else:
        st.write("I don't know this class! It's an **Unknown** class!")
