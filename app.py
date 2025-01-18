import streamlit as st
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from numpy.linalg import norm
import os

# Load the pre-trained embeddings and filenames
@st.cache_resource
def load_embeddings_and_filenames():
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
    return embeddings, filenames

embeddings, filenames = load_embeddings_and_filenames()

# Load the pre-trained VGG19 model
@st.cache_resource
def load_vgg19_model():
    from tensorflow.keras.applications import VGG19
    return VGG19(weights='imagenet', include_top=False, pooling='avg')

model = load_vgg19_model()

# Function to preprocess the uploaded image
def preprocess(img_path, model):
    try:
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        result = model.predict(x)
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"❌ Error processing the image: {e}")
        return None

# Streamlit app UI Improvements

# Sidebar for settings
st.sidebar.title("⚙️ Settings")
num_recommendations = st.sidebar.slider("🔢 Number of Recommendations", min_value=3, max_value=12, step=3, value=6)
similarity_metric = st.sidebar.selectbox("📏 Similarity Metric", ["Euclidean", "Cosine"])
if st.sidebar.button("🔄 Reset"):
    st.experimental_rerun()

# Main UI
st.markdown("""
    <style>
    .stApp {
        background-color: #2F2F2F;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
    }
    .stTextInput>div>input {
        font-size: 18px;
    }
    .stMarkdown {
        color: #E0E0E0;
    }
    .stTitle {
        color: #E0E0E0;
    }
    .stImage>img {
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("👗 Fashion Recommender System")
st.write("📸 Upload an image of your clothing to get recommendations for similar items. 👚")

# File uploader
uploaded_file = st.file_uploader("🔽 Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.markdown("### 👀 Uploaded Image Preview")
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    with st.spinner("🔄 Processing your image..."):
        user_embedding = preprocess(file_path, model)

    if user_embedding is not None:
        # Ensure user_embedding is reshaped for NearestNeighbors
        user_embedding = user_embedding.reshape(1, -1)

        # Find similar items using NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=num_recommendations, metric='euclidean' if similarity_metric == "Euclidean" else 'cosine')
        neighbors.fit(np.array(embeddings).reshape(len(embeddings), -1))
        distances, indices = neighbors.kneighbors(user_embedding)

        # Display recommendations in a grid layout
        st.markdown("### 🛍️ Recommended Similar Items:")

        cols = st.columns(3)  # Create 3 columns for larger images
        for i, index in enumerate(indices[0]):
            with cols[i % 3]:  # Place each recommendation in a column
                st.image(filenames[index], caption=f"🧥 Recommendation {i + 1}", use_container_width=True)

            # Break to the next row after 3 images
            if (i + 1) % 3 == 0:
                cols = st.columns(3)

    # Clean up the temporary file
    if os.path.exists(file_path):
        os.remove(file_path)

else:
    # Show a warning if no image is uploaded
    st.warning("⚠️ No image uploaded. Please upload an image to get recommendations. 📸")

# Footer with credits and social links
st.markdown("""
    <hr>
    <footer style="text-align: center; font-size: 12px; color: #E0E0E0;">
        <p>Developed by [Your Name](https://github.com/Akshat-Sharma-110011). <br> 
        <a href="https://www.linkedin.com/in/akshat-sharma-b71117344/" target="_blank" style="color: #1E90FF;">Connect on LinkedIn</a></p>
    </footer>
""", unsafe_allow_html=True)
