import re


class AudioFile:
    def __init__(self, filename, speaker_name):
        self.speaker_name: str = speaker_name
        self.tonnetz: bool = False
        self.speccontrast: bool = False
        self.chroma: bool = False
        self.mfcc_val: int = 0
        self.mfcc: bool = False
        self.denoised: bool = False
        self.norm_val: float = 0.0
        self.normalized: bool = False
        self.balanced: bool = False
        self.filename = filename
        self.parse_filename()

    def parse_filename(self) -> None:
        pattern = r'(?P<filename>.*?)_(?P<balanced>balanced)?_?(?P<normalized>normalized\((?P<norm_val>-?\d+(\.\d+)?)\))?_?(?P<denoised>denoised)?_?(?P<mfcc>mfcc\((?P<mfcc_val>\d+)\))?_?(?P<chroma>chroma)?_?(?P<speccontrast>speccontrast)?_?(?P<tonnetz>tonnetz)?\.npy'
        match = re.match(pattern, self.filename)
        if match:
            self.filename = match.group('filename')
            self.balanced = bool(match.group('balanced'))
            self.normalized = bool(match.group('normalized'))
            self.norm_val = float(match.group('norm_val')) if match.group('norm_val') else None
            self.denoised = bool(match.group('denoised'))
            self.mfcc = bool(match.group('mfcc'))
            self.mfcc_val = int(match.group('mfcc_val')) if match.group('mfcc_val') else None
            self.chroma = bool(match.group('chroma'))
            self.speccontrast = bool(match.group('speccontrast'))
            self.tonnetz = bool(match.group('tonnetz'))
        else:
            raise ValueError(f"Invalid filename format: {self.filename}")

    def generate_filename(self) -> str:
        file_name: str = self.filename
        if self.balanced:
            file_name += '_balanced'
        if self.normalized:
            file_name += f'_normalized({self.norm_val})'
        if self.denoised:
            file_name += '_denoised'
        if self.mfcc:
            file_name += f'_mfcc({self.mfcc_val})'
        if self.chroma:
            file_name += '_chroma'
        if self.speccontrast:
            file_name += '_speccontrast'
        if self.tonnetz:
            file_name += '_tonnetz'
        file_name += ".npy"

        return file_name
