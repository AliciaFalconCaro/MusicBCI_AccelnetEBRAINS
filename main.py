import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("localhost", 3000))

while True:
    # Receive data (up to 1024 bytes) and the sender's address
    data, addr = sock.recvfrom(1024)
    print(data)