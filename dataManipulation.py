import os
import subprocess
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed

# SETTINGS
INPUT_FOLDER = '/home/predrag/Downloads/fart_model/farts/fart_dataset'
OUTPUT_FOLDER = 'farts'
DURATION_LOG = 'fart.txt'
MAX_WORKERS = 8  # Adjust this to match your CPU cores
MIN_DURATION_SECONDS = 0.5  # Ignore files shorter than this after processing

# FUNCTION to process one file
def process_file(filename):
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    try:
        # Only trim silence (no denoise, no filtering)
        cmd = [
            'ffmpeg',
            '-y',
            '-i', input_path,
            '-af', 'silenceremove=start_periods=1:start_silence=0.05:start_threshold=-20dB:stop_periods=1:stop_silence=0.05:stop_threshold=-20dB',
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Measure new duration
        with wave.open(output_path, 'rb') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration_seconds = frames / float(rate)

        if duration_seconds < MIN_DURATION_SECONDS:
            os.remove(output_path)  # optionally delete short file
            return None  # skip logging

        return f"{filename}: {duration_seconds:.2f} seconds\n"

    except Exception as e:
        return f"{filename}: ERROR ({str(e)})\n"

# MAIN ENTRY POINT
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    file_list = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.wav')]
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_file, filename) for filename in file_list]
        for future in as_completed(futures):
            result = future.result()
            if result:  # Only log if not None
                results.append(result)

    with open(DURATION_LOG, 'w') as log_file:
        log_file.writelines(results)

    print(f"Finished processing {len(results)} valid fart files.")
