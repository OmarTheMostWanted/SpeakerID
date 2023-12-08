import os
import configparser
from cryptography.fernet import Fernet


def generate_key() -> None:
    """
    Generates a key and save it into a file
    """
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)


def load_key() -> bytes:
    """
    Loads the key from the current directory named `secret.key`
    """
    return open("secret.key", "rb").read()


def encrypt_passwords(config: configparser.ConfigParser) -> None:
    """
    Encrypts the passwords in the config
    """
    key = load_key()
    cipher_suite = Fernet(key)
    for section in config.sections():
        for key in config[section]:
            if 'password' in key and not config[section][key].startswith('gAAAAA'):
                password = config[section][key].encode()
                cipher_text = cipher_suite.encrypt(password)
                config[section][key] = cipher_text.decode()


def decrypt_passwords(config: configparser.ConfigParser) -> None:
    """
    Decrypts the passwords in the config
    """
    key = load_key()
    cipher_suite = Fernet(key)
    for section in config.sections():
        for key in config[section]:
            if 'password' in key:
                cipher_text = config[section][key].encode()
                password = cipher_suite.decrypt(cipher_text)
                config[section][key] = password.decode()


def create_config() -> None:
    """
    Creates a config file with default values
    """
    config = configparser.ConfigParser()

    # Add default values
    config.add_section('Paths')
    config.set('Paths', 'Raw Files', '')  # string
    config.set('Paths', 'Wav Files', '')  # string
    config.set('Paths', 'Remove Silence Files', '')  # string
    config.set('Paths', 'Split Files', '')  # string
    config.set('Paths', 'Balanced Files', '')  # string
    config.set('Paths', 'Normalized Files', '')  # string
    config.set('Paths', 'Denoised Files', '')  # string
    config.set('Paths', 'Training Data', '')  # string
    config.set('Paths', 'Models', '')  # string
    config.set("Paths", "Feature Data", "")  # string

    config.add_section('Settings')
    config.set('Settings', 'Convert To wav', 'yes')  # bool
    config.set("Settings", "Remove Silence", 'yes')  # bool
    config.set("Settings", "Split Files", 'yes')  # bool
    config.set('Settings', 'Balance', 'yes')  # bool
    config.set('Settings', 'Normalize', 'yes')  # bool
    config.set('Settings', 'Reduce Noise', 'yes')  # bool
    config.set('Settings', 'N MFCC', '40')  # float
    config.set("Settings", "split minutes", '5')

    config.set('Settings', 'Target Amplitude', '0.0')  # float
    config.set('Settings', 'Average Amplitude', 'yes')  # bool
    config.set("Settings", "Device", "cuda")  # string
    config.set("Settings", "Chunk Size", "100000")  # int
    config.set("Settings", "Overwrite Data", "no")  # bool
    config.set("Settings", "audio duration", '-1')  # bool

    config.add_section('Features')
    config.set('Features', 'MFCC', 'no')  # bool
    config.set('Features', 'Chroma', 'no')  # bool
    config.set('Features', 'Spec Contrast', 'yes')  # bool
    config.set('Features', 'Tonnetz', 'no')  # bool

    # Add default values
    config.add_section('Logging')
    config.set('Logging', 'Url', '127.0.0.1')  # string
    config.set('Logging', 'Port', '5006')  # int
    config.set('Logging', 'App Name', 'App')  # string

    config.add_section('Database')
    config.set('Database', 'Server', '')  # string
    config.set('Database', 'Database', '')  # string
    config.set('Database', 'User', '')  # string
    config.set('Database', 'Password', '')  # string

    config.add_section('DatabaseDebug')
    config.set('DatabaseDebug', 'Server', '')  # string
    config.set('DatabaseDebug', 'Database', '')  # string
    config.set('DatabaseDebug', 'User', '')  # string
    config.set('DatabaseDebug', 'Password', '')  # string

    # Writing configuration to a file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


def read_config() -> configparser.ConfigParser:
    if not os.path.exists('secret.key'):
        generate_key()

    if os.path.exists('config.ini'):
        config = configparser.ConfigParser()
        config.read('config.ini')
        encrypt_passwords(config)
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        decrypt_passwords(config)
        return config
    else:
        create_config()
        print("Config file created. Please fill in the required values.")
        exit()


class Config:
    def __init__(self, config):
        for section in config.sections():
            setattr(self, section, self.Section(config, section))

    class Section:
        def __init__(self, config, section):
            self.__dict__.update(config[section])


def generate_config_class():
    # Read the config file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Start the class definition
    class_def = "class Config:\n"

    # For each section in the config file
    for section in config.sections():
        # Start a subclass definition
        class_def += f"    class {section}:\n"

        # For each key in the section
        for key, value in config[section].items():
            # Add an attribute to the subclass
            if value.isdigit():
                class_def += f"        {key}: int = {value}\n"
            elif value.replace('.', '', 1).isdigit():
                class_def += f"        {key}: float = {value}\n"
            elif value.lower() in ['true', 'false']:
                class_def += f"        {key}: bool = {value.lower() == 'true'}\n"
            else:
                class_def += f"        {key}: str = '{value}'\n"

    # Return the class definition
    return class_def


def generate_config_class2() -> str:
    # Create the config
    create_config()
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Start generating the Config class
    class_code = "class Config:\n"
    for section in config.sections():
        class_code += f"    class {section}:\n"
        for key, value in config.items(section):
            if value.isdigit():
                class_code += f"        {key}: int = {value}\n"
            elif value.replace('.', '', 1).isdigit():
                class_code += f"        {key}: float = {value}\n"
            elif value.lower() in ['yes', 'no']:
                class_code += f"        {key}: bool = {'True' if value.lower() == 'yes' else 'False'}\n"
            else:
                class_code += f"        {key}: str = '{value}'\n"
    return class_code


def create_config_instance(config_class_str: str):
    # Execute the source code to define the Config class
    exec(config_class_str)

    # Create an instance of the Config class
    config_instance = Config()

    # Read the values from the config.ini file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Assign the values to the corresponding attributes in the Config instance
    for section in config.sections():
        for key, value in config.items(section):
            setattr(getattr(config_instance, section), key, value)

    return config_instance


def get_config_instance_dynamic():
    return Config(read_config())


if __name__ == "__main__":
    source_code = generate_config_class2()
    with open("Config.py", 'a') as file:
        file.write(source_code)
