# ------------------------------------------------------
# 🩺 Skin Cancer Detection using VGG16, CapsNet, ResNet
# Streamlit App — Final Version (Hidha)
# ------------------------------------------------------

import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image

# ------------------------------------------------------
# 🧩 Define custom Capsule Network layers (needed for CapsNet)
# ------------------------------------------------------
class SquashLayer(layers.Layer):
    def call(self, vectors, axis=-1):
        s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + 1e-9)
        return scale * vectors

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=[self.num_capsules, input_shape[-1], self.dim_capsule],
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsules, 1, 1])
        inputs_hat = tf.matmul(inputs_tiled, self.W)
        b = tf.zeros_like(inputs_hat[..., 0])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = SquashLayer()(tf.reduce_sum(c[..., None] * inputs_hat, axis=2))
            if i < self.routings - 1:
                b += tf.reduce_sum(inputs_hat * outputs[..., None], axis=-1)
        return outputs


# ------------------------------------------------------
# 📦 Load trained models (cached to avoid reload)
# ------------------------------------------------------
@st.cache_resource
def load_models():
    vgg_model = tf.keras.models.load_model('vgg_functional.keras')
    caps_model = tf.keras.models.load_model(
        'caps_model.keras',
        custom_objects={"SquashLayer": SquashLayer, "CapsuleLayer": CapsuleLayer}
    )
    resnet_model = tf.keras.models.load_model('resnet_final.keras')


    return vgg_model, caps_model, resnet_model


# ------------------------------------------------------
# 🎯 Prediction function
# ------------------------------------------------------
def predict_image(image, vgg_model, caps_model, resnet_model):
    # Convert to array once
    img_array = np.array(image) / 255.0

    # Prepare input for each model
    vgg_img = np.expand_dims(tf.image.resize(img_array, (224, 224)), axis=0)
    resnet_img = np.expand_dims(tf.image.resize(img_array, (224, 224)), axis=0)
    caps_img = np.expand_dims(tf.image.resize(img_array, (64, 64)), axis=0)

    # Get predictions from each model
    vgg_pred = vgg_model.predict(vgg_img)[0][0]
    caps_pred = caps_model.predict(caps_img)[0][0]
    res_pred = resnet_model.predict(resnet_img)[0][0]

    # Combine the predictions (ensemble average)
    avg_pred = np.mean([vgg_pred, caps_pred, res_pred])
    label = "Malignant" if avg_pred > 0.3 else "Benign"



    result = {
        "VGG16": float(vgg_pred),
        "CapsNet": float(caps_pred),
        "ResNet": float(res_pred),
        "Final": float(avg_pred),
        "Label": label
    }

    print("🔍 VGG16:", vgg_pred, "CapsNet:", caps_pred, "ResNet:", res_pred, "Final:", avg_pred)

    return result



# ------------------------------------------------------
# 🖼️ Streamlit Interface
# ------------------------------------------------------
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")
st.title("🩺 Skin Cancer Detection using Deep Learning Ensemble")
st.write("Upload a skin lesion image to classify it as **Benign** or **Malignant**.")

# Load models
with st.spinner("Loading models... please wait ⏳"):
    vgg_model, caps_model, resnet_model = load_models()
st.success("✅ All models loaded successfully!")

# Upload section
uploaded_file = st.file_uploader("📤 Upload a skin image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # 🔍 Debugging info (to check if file is valid)
    st.write("Uploaded File Type:", uploaded_file.type)
    st.write("File Size:", uploaded_file.size)

    try:
        # Try reading directly (Streamlit's safest method)
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_container_width=True)

    except Exception as e:
        st.error("❌ Failed to open image. The file may be corrupted or not a real JPG/PNG.")
        st.error(e)
        st.stop()  # Stop the app so predict button doesn't run


    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            preds = predict_image(image, vgg_model, caps_model, resnet_model)

        st.write("### 🧾 Model Predictions:")
        st.progress(float(preds["Final"]))
        st.write(f"**VGG16 Prediction:** {preds['VGG16']:.4f}")
        st.write(f"**CapsNet Prediction:** {preds['CapsNet']:.4f}")
        st.write(f"**ResNet Prediction:** {preds['ResNet']:.4f}")

        st.subheader(f"🩺 Final Classification: **{preds['Label']}**")

        if preds["Label"] == "Malignant":
            st.error("⚠️ The lesion may be malignant. Please consult a dermatologist.")
        else:
            st.success("✅ The lesion appears benign.")



# ------------------------------------------------------
# ✨ Footer
# ------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Developed by <b>Hidha Subair</b> | Ensemble of VGG16, CapsNet, and ResNet50</p>",
    unsafe_allow_html=True,
)
