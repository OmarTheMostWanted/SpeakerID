import socket


def log(log_text: str, app_name: str, host: str, port: int):
    max_message_length = 99
    log_text = log_text.replace("\r", "")
    total_len = len(log_text)
    next_pos = 0

    while total_len > 0:
        length = min(total_len, max_message_length)
        message = log_text[next_pos:next_pos + length]
        next_pos += length
        total_len -= length

        app_name_bytes = app_name.encode()
        log_text_bytes = message.encode()

        total_length = len(app_name_bytes) + len(log_text_bytes) + 10

        # Create the byte array manually
        udp_packet = bytearray()
        udp_packet.append(0xA1)  # First tag
        udp_packet.append(total_length)  # Total length
        udp_packet.append(0x81)  # Second tag for app name
        udp_packet.append(len(app_name_bytes))  # Length of app name
        udp_packet.extend(app_name_bytes)  # App name bytes
        udp_packet.append(0x82)  # Third tag for log text
        udp_packet.append(len(log_text_bytes))  # Length of log text
        udp_packet.extend(log_text_bytes)  # Log text bytes
        udp_packet.append(0x83)  # Fourth tag for log number
        udp_packet.append(4)  # Length of log number (4 bytes)
        udp_packet.extend(bytes([0, 0, 0, 0]))  # Log number (always 0)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
            udp_socket.sendto(udp_packet, (host, port))
