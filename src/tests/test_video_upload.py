import cv2
import socket


# Function to send frame over socket connection
def send_frame(sock, frame):
    if frame is None:
        # Signal end of video
        sock.sendall(b'END')
    else:
        # Convert frame to bytes
        frame_bytes = frame.tobytes()
        # Get the size of the frame in bytes
        size = len(frame_bytes)
        # Send the size of the frame to the server
        sock.sendall(size.to_bytes(8, byteorder='big'))
        # Send the frame data to the server
        sock.sendall(frame_bytes)


# Open video file
video = cv2.VideoCapture('test.mp4')
# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Define the server address and port
server_address = ('localhost', 12345)
# Connect to the server
sock.connect(server_address)
# Read and send each frame of the video
while True:
    # Read the frame
    ret, frame = video.read()
    # Check if frame was read successfully
    if not ret:
        break
    # Send the frame over the socket connection
    send_frame(sock, frame)
# Signal the end of the video by sending an empty frame
# send_frame(sock, None)
# Close the video file and socket connection
video.release()
sock.close()
