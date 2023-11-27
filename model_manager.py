import re

class ModelFile:
    def __init__(self, model_name):
        self.accuracy: float = 0.0
        self.speakers: list = []
        self.balanced: bool = False
        self.normalized: bool = False
        self.norm_val: float = 0.0
        self.denoised: bool = False
        self.mfcc: bool = False
        self.mfcc_val: int = 0
        self.chroma: bool = False
        self.speccontrast: bool = False
        self.tonnetz: bool = False
        self.model_name = model_name
        self.parse_model_name()

    def __init__(self,):
        self.accuracy: float = 0.0
        self.speakers: list = []
        self.balanced: bool = False
        self.normalized: bool = False
        self.norm_val: float = 0.0
        self.denoised: bool = False
        self.mfcc: bool = False
        self.mfcc_val: int = 0
        self.chroma: bool = False
        self.speccontrast: bool = False
        self.tonnetz: bool = False
        self.model_name = ''

    def parse_model_name(self) -> None:
        pattern = r'model\((?P<accuracy>\d+\.\d+)\)\[(?P<speakers>.+?)\]_?(?P<balanced>balanced)?_?(?P<normalized>normalized\((?P<norm_val>-?\d+(\.\d+)?)\))?_?(?P<denoised>denoised)?_?(?P<mfcc>mfcc\((?P<mfcc_val>\d+)\))?_?(?P<chroma>chroma)?_?(?P<speccontrast>speccontrast)?_?(?P<tonnetz>tonnetz)?\.plk'
        match = re.match(pattern, self.model_name)
        if match:
            self.accuracy = float(match.group('accuracy'))
            self.speakers = [speaker.strip() for speaker in match.group('speakers').split(',')]
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
            raise ValueError(f"Invalid model name format: {self.model_name}")

    def generate_model_name(self) -> str:
        model_name = f"model({self.accuracy})["
        model_name += ', '.join(self.speakers) + "]"
        if self.balanced:
            model_name += '_balanced'
        if self.normalized:
            model_name += f'_normalized({self.norm_val})'
        if self.denoised:
            model_name += '_denoised'
        if self.mfcc:
            model_name += f'_mfcc({self.mfcc_val})'
        if self.chroma:
            model_name += '_chroma'
        if self.speccontrast:
            model_name += '_speccontrast'
        if self.tonnetz:
            model_name += '_tonnetz'
        model_name += ".plk"

        return model_name

class LabelEncoderFile:
    def __init__(self, le_name):
        self.accuracy: float = 0.0
        self.speakers: list = []
        self.balanced: bool = False
        self.normalized: bool = False
        self.norm_val: float = 0.0
        self.denoised: bool = False
        self.mfcc: bool = False
        self.mfcc_val: int = 0
        self.chroma: bool = False
        self.speccontrast: bool = False
        self.tonnetz: bool = False
        self.le_name = le_name
        self.parse_model_name()

    def __init__(self,):
        self.accuracy: float = 0.0
        self.speakers: list = []
        self.balanced: bool = False
        self.normalized: bool = False
        self.norm_val: float = 0.0
        self.denoised: bool = False
        self.mfcc: bool = False
        self.mfcc_val: int = 0
        self.chroma: bool = False
        self.speccontrast: bool = False
        self.tonnetz: bool = False
        self.le_name = ''

    def parse_model_name(self) -> None:
        pattern = r'label_encoder\((?P<accuracy>\d+\.\d+)\)\[(?P<speakers>.+?)\]_?(?P<balanced>balanced)?_?(?P<normalized>normalized\((?P<norm_val>-?\d+(\.\d+)?)\))?_?(?P<denoised>denoised)?_?(?P<mfcc>mfcc\((?P<mfcc_val>\d+)\))?_?(?P<chroma>chroma)?_?(?P<speccontrast>speccontrast)?_?(?P<tonnetz>tonnetz)?\.plk'
        match = re.match(pattern, self.le_name)
        if match:
            self.accuracy = float(match.group('accuracy'))
            self.speakers = [speaker.strip() for speaker in match.group('speakers').split(',')]
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
            raise ValueError(f"Invalid label encoder name format: {self.le_name}")

    def generate_le_name(self) -> str:
        le_name = f"label_encoder({self.accuracy})["
        le_name += ', '.join(self.speakers) + "]"
        if self.balanced:
            le_name += '_balanced'
        if self.normalized:
            le_name += f'_normalized({self.norm_val})'
        if self.denoised:
            le_name += '_denoised'
        if self.mfcc:
            le_name += f'_mfcc({self.mfcc_val})'
        if self.chroma:
            le_name += '_chroma'
        if self.speccontrast:
            le_name += '_speccontrast'
        if self.tonnetz:
            le_name += '_tonnetz'
        le_name += ".plk"

        return le_name
