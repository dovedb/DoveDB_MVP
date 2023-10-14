# client.py

import socket

HOST = '127.0.0.1'
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    
    while True:
        query = input(">>")
        
        if query.lower() == 'exit':
            break
        
        s.sendall(query.encode('utf-8'))
        
        data = s.recv(1024)
        print(data.decode('utf-8'))

print("Connection closed.")
