import sys

test_dir = "/home/tmw/Code/ML-Data/test dir"

mp3_file = "/home/tmw/Code/ML-Data/test dir/antietambattle_02_tilberg_64kb.mp3"

import audio_tools as at

if "__name__" == "__main__":

    if len(sys.argv) == 1:



        at.convert_file_to_wav(mp3_file, wav_path="temp.wav")

