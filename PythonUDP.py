import socket
import sys

UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 9900

# Create a UDP socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Bind the socket to the port
server_address = (UDP_IP_ADDRESS, UDP_PORT_NO)
s.bind(server_address)
print("Do Ctrl+c to exit the program !!")

while True:
    #s.sendto(send_data.encode('utf-8'), (UDP_IP_ADDRESS, UDP_PORT_NO))
    s.sendto(bytes("test", 'utf-8'), (UDP_IP_ADDRESS, UDP_PORT_NO))
    #print("\n\n 1. Server sent : ", "test","\n\n")
