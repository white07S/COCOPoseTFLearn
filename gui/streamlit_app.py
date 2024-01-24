import sys
sys.path.append('/home/preetam/projects/project2/COCOPoseTFLearn')
import streamlit as st
import cv2
import numpy as np
from model import cmu_model
from process_image import process_image
import visualize_parts  # Correct the spelling if it's different

def main():
    st.title('Pose Estimation App')

    # Load model
    model = cmu_model.get_testing_model()
    model.load_weights('weights.best.h5')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        oriImg = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Store the original image in the session state
        st.session_state['oriImg'] = oriImg

        st.image(oriImg[:, :, [2, 1, 0]], caption='Uploaded Image.', use_column_width=True)

        if 'processed_images' not in st.session_state:
            st.session_state['processed_images'] = []

        if 'process_button_clicked' not in st.session_state:
            if st.button('Process Image'):
                st.session_state['process_button_clicked'] = True
                # Process image
                fig1, fig2, fig3, paf_avg, heatmap_avg = process_image(oriImg, model)

                # Store the results in Streamlit session to use in next steps
                st.session_state['paf_avg'] = paf_avg
                st.session_state['heatmap_avg'] = heatmap_avg

                # Display and store results
                st.session_state['processed_images'].append(fig1)
                st.session_state['processed_images'].append(fig2)
                st.session_state['processed_images'].append(fig3)

        if 'process_button_clicked' in st.session_state:
            for fig in st.session_state['processed_images']:
                st.pyplot(fig)

        if 'visualize_button_clicked' not in st.session_state and 'paf_avg' in st.session_state and 'heatmap_avg' in st.session_state:
            if st.button('Visualize Body Parts'):
                st.session_state['visualize_button_clicked'] = True
                to_plot, subset, candidate = visualize_parts.visualize_parts_mapping(
                    st.session_state['oriImg'], 
                    st.session_state['paf_avg'], 
                    st.session_state['heatmap_avg']
                )
                st.session_state['subset'] = subset
                st.session_state['candidate'] = candidate
                st.session_state['visualized'] = to_plot

        if 'visualize_button_clicked' in st.session_state:
            st.image(st.session_state['visualized'][:, :, [2, 1, 0]], caption='Visualized Body Parts', use_column_width=True)

        if 'link_points_button_clicked' not in st.session_state and 'visualized' in st.session_state:
            if st.button('Link Points'):
                st.session_state['link_points_button_clicked'] = True
                linked_image = visualize_parts.link_points(
                    st.session_state['oriImg'].copy(), 
                    st.session_state['subset'], 
                    st.session_state['candidate']
                )
                st.session_state['joints_linked'] = linked_image

        if 'link_points_button_clicked' in st.session_state:
            st.image(st.session_state['joints_linked'][:, :, [2, 1, 0]], caption='Joint Plot', use_column_width=True)

if __name__ == '__main__':
    main()
