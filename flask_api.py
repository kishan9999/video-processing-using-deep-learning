from flask import Flask, render_template, request
import cv2
import os
import tempfile
import logging

app = Flask(__name__)

# Temporary folder to store uploaded files
UPLOAD_FOLDER = tempfile.gettempdir()

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Adjust logging level as needed

def get_video_info2(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        return None

    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration_sec = frame_count / fps
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video object
    video.release()

    return {
        'duration_sec': f"{duration_sec:.2f} seconds",
        'fps': fps,
        'resolution': f"{width} x {height}",
        'total_frames': frame_count
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    video_info = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        try:
            # Save the file to a temporary location
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Print uploaded filename
            logging.debug(f"Uploaded filename: {file.filename}")

            # Get video information
            video_info = get_video_info2(file_path)

            if video_info:
                logging.debug(f"Video info: {video_info}")
            else:
                logging.debug("Error: Could not open video file")

        except Exception as e:
            logging.error(f"Error processing video: {e}")
            return render_template('index.html', error='Error processing video')

        finally:
            # Remove the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

    return render_template('index.html', video_info=video_info)

if __name__ == '__main__':
    app.run(use_reloader=False, debug=True, host='0.0.0.0', port=81)
