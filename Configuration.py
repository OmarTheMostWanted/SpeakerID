import configparser
from cryptography.fernet import Fernet

def generate_key():
    return Fernet.generate_key()

def init_cipher(key):
    return Fernet(key)

def encrypt_value(cipher, value):
    return cipher.encrypt(value.encode()).decode()

def decrypt_value(cipher, encrypted_value):
    return cipher.decrypt(encrypted_value.encode()).decode()

def read_config(file_path, key=None):
    config = configparser.ConfigParser()

    # Check if the file exists
    if not config.read(file_path):
        print(f"Config file not found at {file_path}. Generating a template.")
        generate_template(file_path, key)
        config.read(file_path)
    else:
        # Decrypt passwords when the config file already exists
        decrypt_passwords(config, key)

    return config

def generate_template(file_path, key=None):
    # Create a template configuration
    template_config = configparser.ConfigParser()

    template_config['Database'] = {
        'host': '',
        'port': '',
        'username': '',
        'password': encrypt_value(key, '') if key else ''
    }

    template_config['Paths'] = {
        'data_path': '',
        'log_path': ''
    }

    # Write the template to the file
    with open(file_path, 'w') as config_file:
        template_config.write(config_file)

def encrypt_passwords(config, key):
    cipher = init_cipher(key)

    for section in config.sections():
        for key, value in config.items(section):
            if key.lower() == 'password':
                config[section][key] = encrypt_value(cipher, value)

if __name__ == "__main__":
    # Example usage
    config_path = 'config.ini'
    encryption_key = generate_key()

    config = read_config(config_path, key=encryption_key)

    # Access values from the configuration
    database_host = config['Database']['host']
    database_port = config['Database']['port']
    database_username = config['Database']['username']
    database_password = config['Database']['password']

    data_path = config['Paths']['data_path']
    log_path = config['Paths']['log_path']

    # Use the retrieved values as needed
    print(f"Database Host: {database_host}")
    print(f"Data Path: {data_path}")

    # Check if passwords need encryption
    if not database_password:
        print("Passwords are not encrypted. Encrypting now.")
        encrypt_passwords(config, encryption_key)

        # Save the encrypted passwords to the config file
        with open(config_path, 'w') as config_file:
            config.write(config_file)
