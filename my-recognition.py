import os
import jetson_inference
import jetson_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--model", type=str, required=True, help="path to your custom model (.onnx or .engine)")
parser.add_argument("--labels", type=str, required=True, help="path to your labels.txt file")
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)

# Load the image classification model
net = jetson_inference.imageNet(
    argv=[
        "--model=" + opt.model,
        "--labels=" + opt.labels,
        "--input_blob=input_0",
        "--output_blob=output_0",
        "--verbose"
    ]
)

class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)

print(f"image is recognized as {class_desc} (class #{class_idx}) with {confidence*100:.2f}% confidence")

# Provide advice based on the recognized emotion
advice = {
    "angry": "Try to take deep breaths and step away from the situation for a moment.",
    "disgust": "Focus on something positive or take a break to clear your mind.",
    "fear": "Remember, it's okay to feel afraid. Talk to someone you trust or practice calming techniques.",
    "happy": "Keep enjoying the moment and share your happiness with others!",
    "neutral": "Stay balanced and keep moving forward.",
    "sad": "Reach out to a friend or do something you love to lift your spirits.",
    "surprise": "Embrace the unexpected and see what you can learn from it."
}

print(f"Advice: {advice.get(class_desc, 'No advice available for this emotion.')}")

from textwrap import wrap

font = jetson_utils.cudaFont()

emotion_text = f"Emotion: {class_desc} ({confidence*100:.1f}%)"
advice_raw = advice.get(class_desc, "No advice available.")

# 1. Set smaller wrap width for narrow images
max_chars_per_line = max(int(img.width / 18), 15)  # tighter wrap
advice_lines = wrap(advice_raw, width=max_chars_per_line)

# 2. Set smaller line height to fit more vertically
line_height = 26
padding = 10
total_lines = 1 + len(advice_lines)
needed_height = total_lines * line_height + 2 * padding

# 3. Crop text if vertical space is still not enough
max_lines = (img.height - 2 * padding) // line_height - 1
if len(advice_lines) > max_lines:
    advice_lines = advice_lines[:max_lines]
    advice_lines[-1] += "..."  # indicate text was trimmed
    total_lines = 1 + len(advice_lines)
    needed_height = total_lines * line_height + 2 * padding

# 4. Draw background box
jetson_utils.cudaDrawRect(
    img,
    (0, 0, img.width, needed_height),
    (0, 0, 0, 230)  # dark background
)

# 5. Draw text line by line
font.OverlayText(img, img.width, img.height, emotion_text, 10, 10, (0, 255, 0, 255))
for i, line in enumerate(advice_lines):
    y = 10 + line_height * (i + 1)
    font.OverlayText(img, img.width, img.height, line, 10, y, (255, 255, 0, 255))

# Save image
results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
output_filename = os.path.join(results_dir, f"{class_desc}_{os.path.basename(opt.filename)}")
jetson_utils.saveImage(output_filename, img)
print(f"Image exported to {output_filename}")

