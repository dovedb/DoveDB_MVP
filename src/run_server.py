import socket
from vmql_transformer import process_vmql


def run_video_server():
    from server.video_server import VideoServer
    video_server = VideoServer()
    print('Run video server...')
    video_server.listen()


def run_db_server():
    HOST = '127.0.0.1'
    PORT = 65432

    def process_query(query):
        print('try to process query: ', query)
        response = process_vmql(query)
        return response

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print('Server is listening...')
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                # Decode received data and process query
                response = process_query(data.decode('utf-8'))
                # Send the response back to the client
                conn.sendall(response.encode('utf-8'))
    conn.close()

if __name__ == '__main__':
    # run_video_server()
    run_db_server()
