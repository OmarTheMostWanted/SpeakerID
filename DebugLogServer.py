import socket

def udp_listener(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        udp_socket.bind((host, port))
        print(f"Listening for UDP packets on {host}:{port}...")
        while True:
            data, addr = udp_socket.recvfrom(1024)  # Adjust the buffer size as needed
            parse_and_display_log(data)

def parse_and_display_log(data):
    try:
        # Assuming the ASN.1 structure: [0xA1, total_length, 0x81, app_name_length, app_name, 0x82, log_text_length, log_text, 0x83, 0x04, log_number]
        tag_index = 0
        app_name_length = data[3]
        app_name = data[4:4 + app_name_length].decode()
        log_text_length = data[5 + app_name_length]
        log_text = data[6 + app_name_length:6 + app_name_length + log_text_length].decode()
        log_number = int.from_bytes(data[7 + app_name_length + log_text_length:11 + app_name_length + log_text_length], byteorder='big')

        formatted_log = f"{str(log_number).zfill(6)},{app_name}, {log_text}"
        print(f"Received log message: {formatted_log}")
    except Exception as e:
        print(f"Error parsing log message: {e}")

# Example usage:
host = "0.0.0.0"  # Listen on all available network interfaces
port = 5006     # Should match the port used in your sender script
udp_listener(host, port)
