# Prediction interface for Cog ⚙️
# https://cog.run/python

import logging
from rembg import remove, new_session
import os
from multiprocessing import Process


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

output_file = "output.webm"
frame_rate = 30
input_frames_dir = "./input_frames"
output_frames_dir = "./output_frames"

session = new_session("u2netp")

cpu_cores = 4


def process_frames_chunk(start_idx, end_idx, frames):
    """Process a chunk of frames using the shared session"""
    for i in range(start_idx, end_idx):
        if i >= len(frames):
            break
            
        frame = frames[i]
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


def remove_background_from_frames():
    frames = os.listdir(input_frames_dir)
    num_frames = len(frames)
    logging.info(f"Processing {num_frames} frames with {cpu_cores} processes")

    # Calculate chunk size for each process
    chunk_size = (num_frames + cpu_cores - 1) // cpu_cores
    
    # Create and start processes
    processes = []
    for i in range(cpu_cores):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_frames)
        
        p = Process(target=process_frames_chunk, 
                   args=(start_idx, end_idx, frames))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    logging.info(f"Processed {num_frames} frames")