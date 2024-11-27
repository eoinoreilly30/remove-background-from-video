# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import logging
from rembg import remove, new_session
import os
import subprocess
import time


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

output_file = "output.webm"
frame_rate = 30
input_frames_dir = "./input_frames"
output_frames_dir = "./output_frames"

session = new_session("u2netp")

def process_frame(frame):
    input_frame_path = f'{input_frames_dir}/{frame}'
    output_frame_path = f'{output_frames_dir}/{frame}'
        
    with open(input_frame_path, 'rb') as i:
        with open(output_frame_path, 'wb') as o:
            logging.info(f"Reading input file {input_frame_path}")
            input_data = i.read()
            
            logging.info(f"Starting remove background process for {input_frame_path}")
            output = remove(input_data, session=session, bgcolor=(255, 255, 255, 0))

            logging.info(f"Writing output file {output_frame_path}")
            o.write(output)
    return frame


def remove_background_from_frames():
    frames = os.listdir(input_frames_dir)
    logging.info(f"Processing {len(frames)} frames sequentially")

    for frame in frames:
        process_frame(frame)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        logging.info("Setting up session")
        os.environ['U2NET_HOME'] = os.path.expanduser('~/.u2net')

        logging.info(f"Checking model exists at {os.environ['U2NET_HOME']}")
        if os.path.exists(os.environ['U2NET_HOME']):
            for file in os.listdir(os.environ['U2NET_HOME']):
                logging.info(f"{file} - {os.path.getsize(os.path.join(os.environ['U2NET_HOME'], file))}b")
        else:
            logging.warning(f"Directory {os.environ['U2NET_HOME']} does not exist")
            
    def predict(
        self,
        input_video: Path = Input(description="Input video to process"),
    ) -> Path:
        """Run a single prediction on the model"""

        start_time = time.time()

        # Create directories
        logging.info("Creating directories")
        input_frames_dir = "./input_frames"
        output_frames_dir = "./output_frames"

        if os.path.exists(input_frames_dir):
            logging.info(f"Removing existing input frames directory {input_frames_dir}")
            for file in os.listdir(input_frames_dir):
                os.remove(os.path.join(input_frames_dir, file))
            os.rmdir(input_frames_dir)
            
        if os.path.exists(output_frames_dir):
            logging.info(f"Removing existing output frames directory {output_frames_dir}")
            for file in os.listdir(output_frames_dir):
                os.remove(os.path.join(output_frames_dir, file))
            os.rmdir(output_frames_dir)
            
        os.makedirs(input_frames_dir)
        os.makedirs(output_frames_dir)
        
        # Split video into frames using ffmpeg
        logging.info("Splitting video into frames")
        frame_name_format = 'frame_%04d.png'

        ffmpeg_cmd = ['ffmpeg', 
                      '-y', 
                      '-i', str(input_video), 
                      '-vf', f'fps={frame_rate}', 
                      '-pix_fmt', 'rgb24',
                      f'{input_frames_dir}/{frame_name_format}']
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Process frames sequentially
        remove_background_from_frames()

        # Reassemble video from frames
        logging.info("Reassembling video from frames")
        ffmpeg_cmd = ['ffmpeg',
                      '-y',
                      '-framerate', f'{frame_rate}',
                      '-i', f'{output_frames_dir}/{frame_name_format}',
                      '-c:v', 'libvpx-vp9',
                      '-pix_fmt', 'yuva420p',
                      '-lossless', '1',
                      '-quality', 'good',
                      '-cpu-used', '0',
                      '-b:v', '2M',
                      output_file]
        
        subprocess.run(ffmpeg_cmd, check=True)

        logging.info(f"Remove background process finished successfully in {time.time() - start_time} seconds")
        return Path(output_file)
