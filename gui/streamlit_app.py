import streamlit as st
import os

# Create a simple Streamlit app with two tabs: Image and Video

def main():
    # Set up the main layout
    st.title("Pose Estimation Demo")

    # Create tabs
    tab1, tab2 = st.tabs(["Image", "Video"])

    # Image Tab
    with tab1:
        st.header("Image Display")
        st.subheader("Upload an Image")
        image_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if image_file is not None:
            st.image(image_file, caption='Uploaded Image.', use_column_width=True)

    # Video Tab
    with tab2:
        st.header("Video Display")
        st.subheader("Upload a Video")
        video_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"], key="video")
        if video_file is not None:
            st.video(video_file)

if __name__ == "__main__":
    main()
