import os
import configparser
from cryptography.fernet import Fernet


def generate_key():
    """
    Generates a key and save it into a file
    """
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)


def load_key():
    """
    Loads the key from the current directory named `secret.key`
    """
    return open("secret.key", "rb").read()


def encrypt_passwords(config):
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


def decrypt_passwords(config):
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


def main():
    if not os.path.exists('secret.key'):
        generate_key()

    if os.path.exists('config.ini'):
        config = configparser.ConfigParser()
        config.read('config.ini')
        encrypt_passwords(config)
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        decrypt_passwords(config)

        print(config["Section1"]["password"])
        # Now you can use the decrypted config
    else:
        create_config()
        print("Config file created. Please fill in the required values.")
        exit()


if __name__ == "__main__":
    main()

# import configparser
# from cryptography.fernet import Fernet
#
# def generate_key():
#     return Fernet.generate_key()
#
# def init_cipher(key):
#     return Fernet(key)
#
# def encrypt_value(cipher, value):
#     return cipher.encrypt(value.encode()).decode()
#
# def decrypt_value(cipher, encrypted_value):
#     return cipher.decrypt(encrypted_value.encode()).decode()
#
# def read_config(file_path, key=None):
#     config = configparser.ConfigParser()
#
#     # Check if the file exists
#     if not config.read(file_path):
#         print(f"Config file not found at {file_path}. Generating a template.")
#         generate_template(file_path, key)
#         config.read(file_path)
#     else:
#         # Decrypt passwords when the config file already exists
#         decrypt_passwords(config, key)
#
#     return config
#
# def generate_template(file_path, key=None):
#     # Create a template configuration
#     template_config = configparser.ConfigParser()
#
#     template_config['Database'] = {
#         'host': '',
#         'port': '',
#         'username': '',
#         'password': encrypt_value(key, '') if key else ''
#     }
#
#     template_config['Paths'] = {
#         'data_path': '',
#         'log_path': ''
#     }
#
#     # Write the template to the file
#     with open(file_path, 'w') as config_file:
#         template_config.write(config_file)
#
# def encrypt_passwords(config, key):
#     cipher = init_cipher(key)
#
#     for section in config.sections():
#         for key, value in config.items(section):
#             if key.lower() == 'password':
#                 config[section][key] = encrypt_value(cipher, value)
#
# if __name__ == "__main__":
#     # Example usage
#     config_path = 'config.ini'
#     encryption_key = generate_key()
#
#     config = read_config(config_path, key=encryption_key)
#
#     # Access values from the configuration
#     database_host = config['Database']['host']
#     database_port = config['Database']['port']
#     database_username = config['Database']['username']
#     database_password = config['Database']['password']
#
#     data_path = config['Paths']['data_path']
#     log_path = config['Paths']['log_path']
#
#     # Use the retrieved values as needed
#     print(f"Database Host: {database_host}")
#     print(f"Data Path: {data_path}")
#
#     # Check if passwords need encryption
#     if not database_password:
#         print("Passwords are not encrypted. Encrypting now.")
#         encrypt_passwords(config, encryption_key)
#
#         # Save the encrypted passwords to the config file
#         with open(config_path, 'w') as config_file:
#             config.write(config_file)
