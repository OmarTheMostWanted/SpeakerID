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
    config.set('Paths', 'Training Data', '')
    config.set('Paths', 'Models', '')

    # Add default values
    config.add_section('Logging')
    config.set('Logging', 'Url', '')
    config.set('Logging', 'Port', '')
    config.set('Logging', 'App Name', '')

    config.add_section('Database')
    config.set('Database', 'Server', '')
    config.set('Database', 'Database Name', '')
    config.set('Database', 'User Name', '')
    config.set('Database', 'Password', '')

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


if __name__ == "__main__":
    read_config()