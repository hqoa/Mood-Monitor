=> **Mood Monitor**

A Jetson Nano-based emotion recognition system that detects human facial emotions from images and offers personalized mental wellness advice. The model uses a custom-trained neural network to classify expressions like happy, sad, angry, fear, etc., and overlays relevant advice on the image automatically.

Mood Monitor Output Sample: (`https://drive.google.com/file/d/1bW4BgVj1Dd19Omwaj4C0nDDlRPXVpRnF/view?usp=sharing`)


=> **The Algorithm**

This project uses NVIDIA Jetson Nano and the `jetson-inference` framework with a custom ONNX image classification model. 

The steps include:
- Load an image using `jetson_utils.loadImage()`
- Use `jetson_inference.imageNet` to classify the emotion
- Retrieve label, confidence, and matching wellness advice from a dictionary
- Overlay emotion label and wrapped advice directly onto the image using `cudaFont.OverlayText()`
- Save the result image with emotion-based filename into a `results/` folder

**Model:** Custom-trained emotion classifier (`.onnx`)  
**Input:** Static image of a face  
**Output:** Image with emotion + advice overlayed


=> **Running this Project**

1. Clone this repository:
    ```bash
    git clone https://github.com/hqoa/Mood-Monitor.git
    cd Mood-Monitor
    ```
2. Import an image of your liking to the images folder and name it `picture.jpg`.

3. Make sure you have these files:
   - `labels.txt` file in the `models` folder with one emotion label per line in the.
   - `model.onnx` file in the `models` folder.

4. Run the script:
    ```bash
    python3 my-recognition.py picture.jpg --model=model.onnx --labels=labels.txt
    ```

5. Check the `results/` folder for output image with text overlay.


=> **Dependencies**
Install the required Jetson utilities (already pre-installed on JetPack environments):

```bash
sudo apt-get install python3-pip
pip3 install jetson-inference jetson-utils
