from server.video_server import VideoServer


if __name__ == '__main__':
    video_server = VideoServer()
    print('Run video server...')
    video_server.listen()
