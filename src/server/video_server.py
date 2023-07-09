import cv2
import socket
import numpy as np


class VideoServer(object):

    def __init__(self):
        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Define the server address and port
        self.server_address = ('localhost', 12345)

    def listen(self):
        # Bind the socket to the server address and port
        self.sock.bind(self.server_address)
        # Listen for incoming connections
        self.sock.listen(1)
        # Accept a client connection
        client_socket, client_address = self.sock.accept()
        # Open a video writer
        count = 0
        while True:
            # Receive the frame from the client
            frame = self.receive_frame(client_socket)
            # If frame is None, it indicates the end of the video
            if frame is None:
                break
            # Resize the frame to half its original size
            resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            # Write the resized frame to the output video
            cv2.imwrite(f'frames/{count}.jpg', resized_frame)
            count += 1
        # Release the video writer and close the sockets
        client_socket.close()
        self.sock.close()

    # Function to receive frame from the client
    def receive_frame(self, sock):

        height = 480
        width = 640

        # Receive the size of the frame
        size_bytes = sock.recv(8)
        size = int.from_bytes(size_bytes, byteorder='big')

        if size > 0:
            # Receive the frame data
            frame_data = b''
            bytes_received = 0
            while bytes_received < size:
                data = sock.recv(size - bytes_received)
                if not data:
                    break
                frame_data += data
                bytes_received += len(data)

            # Convert the frame data to a NumPy array
            frame = np.frombuffer(frame_data, dtype=np.uint8)

            # Reshape the frame array
            frame = frame.reshape((-1, 3))

            # Convert the frame array to an image
            image = frame.reshape((height, width, 3))

            # Resize the image to half its size
            resized_image = cv2.resize(image, (width // 2, height // 2))
            return resized_image

