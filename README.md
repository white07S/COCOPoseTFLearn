# COCOPoseTFLearn


## Installation and Setup Instructions

Follow these steps to set up the project environment:

1. **Initialize Windows Subsystem for Linux (WSL):**
   Launch the Windows Subsystem for Linux on your machine.

2. **Clone the Repository:**
   Using Git, clone the repository and open it in Visual Studio Code.

3. **Prepare the Datasets:**
   - Navigate to the datasets folder:
     ```
     cd datasets
     ```
   - Modify script permissions and download the dataset:
     ```
     chmod +x download_dataset.sh
     ./download_dataset.sh
     ```
   Note: The download process may take approximately 45 minutes. Ensure you have at least 25-30 GB of free space on your WSL instance.

4. **Create a Conda Environment:**
   Set up a new Conda environment named 'cv' with the required dependencies:
   ```
   conda create --name cv --file conda_requirements.txt
   ```

5. **Activate the Conda Environment:**
   Activate the newly created Conda environment:
   ```
   conda activate cv
   ```

6. **Install Additional Python Dependencies:**
   Install the required Python packages using pip:
   ```
   pip install -r requirements.txt
   ```

7. **Start the Training Process:**
   Initiate the training process by running the training script:
   ```
   python training/train_pose.py
   ```

8. **TensorBoard and Model Visualization:**
   During the training, a popup will prompt you to open TensorBoard in Visual Studio Code and provide an option to download PNGs of the model.

