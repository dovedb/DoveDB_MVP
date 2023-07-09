import math
import psycopg2
from psycopg2 import sql
import xml.etree.ElementTree as ET
import os

class OpenGaussManager:

    def __init__(self, dbname, user, password, host, port):
        self.conn = psycopg2.connect(database=dbname, user=user, password=password, host=host, port=port)
        self.cur = self.conn.cursor()

    def create_tables(self):
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS Frame(
                video_id varchar(20),
                frame_number int,
                bbox_id varchar(20),
                PRIMARY KEY (video_id, frame_number,bbox_id)
            );
            CREATE TABLE IF NOT EXISTS Bbox(
                bbox_id varchar(20),
                object_id varchar(20),
                class varchar(20),
                bbox_xmin float,
                bbox_ymin float,
                bbox_xmax float,
                bbox_ymax float,
                PRIMARY KEY (bbox_id)
            );
        ''')
        self.conn.commit()

    def insert_data(self, table, data):
        placeholders = ', '.join(['%s'] * len(data))
        columns = ', '.join(data.keys())
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.cur.execute(sql, list(data.values()))
        self.conn.commit()

    def query_data(self, table, condition):
        placeholders = ' AND '.join([f"{k} = %s" for k in condition.keys()])
        sql = f"SELECT * FROM {table} WHERE {placeholders}"
        self.cur.execute(sql, list(condition.values()))
        return self.cur.fetchall()


    def query_car_id_count(self, cam_id, time_start):
        """
        query the number of unique car_id in the given time interval
        Args:
            video_id (int): the video id of the camera to be queried
            time_start (int): the start time of the time interval
        """
        query = """
            SELECT COUNT(DISTINCT B.object_id)
            FROM Frame F
            JOIN Bbox B ON F.bbox_id = B.bbox_id
            WHERE F.video_id = %s AND F.frame_number > %s 
        """
        self.cur.execute(query,(cam_id, time_start))
        return self.cur.fetchall()

    def query_most_car_frame(self, cam_id, time_start):
        
        query = """
            SELECT F.video_id, F.frame_number, COUNT(*) AS car_count
            FROM Frame F
            JOIN Bbox B ON F.bbox_id = B.bbox_id
            WHERE F.video_id = %s AND F.frame_number > %s
            GROUP BY F.video_id, F.frame_number
            ORDER BY car_count DESC
            LIMIT 10;
        """
        self.cur.execute(query,(cam_id, time_start))
        return self.cur.fetchall()
    
    def update_data(self, table, data, condition):
        set_clause = ', '.join([f"{k} = %s" for k in data.keys()])
        where_clause = ' AND '.join([f"{k} = %s" for k in condition.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        self.cur.execute(sql, list(data.values()) + list(condition.values()))
        self.conn.commit()

    def delete_data(self, table, condition):
        placeholders = ' AND '.join([f"{k} = %s" for k in condition.keys()])
        sql = f"DELETE FROM {table} WHERE {placeholders}"
        self.cur.execute(sql, list(condition.values()))
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()
def test_opengauss_manager():

    # init OpenGaussManager class
    ogm = OpenGaussManager(dbname="test_db", user="gaussdb", password="Secretpassword@123", host="127.0.0.1", port="15432")
    #  delete all tables
    ogm.cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public';")
    tables = ogm.cur.fetchall()

    for table in tables:
        # delete each table
        ogm.cur.execute(sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE;").format(sql.Identifier(table[0])))

    # create_tables()
    ogm.create_tables()

    # inert data into Frame table
    data = {
        'video_id': 'video_1',
        'frame_number': 1,
        'bbox_id': 'bbox_1'
    }
    ogm.insert_data('Frame', data)

     # insert data into Bbox table
    bbox_data = {
        'bbox_id': 'bbox_1',
        'object_id': 'object_1',
        'class': 'car',
        'bbox_xmin': 1.0,
        'bbox_ymin': 1.0,
        'bbox_xmax': 1.0,
        'bbox_ymax': 1.0
    }
    ogm.insert_data('Bbox',bbox_data)

    
    
    # query data
    condition = {'video_id': 'video_1'}
    print(ogm.query_data('Frame', condition))

    condition2 = {'bbox_id': 'bbox_1'}
    print(ogm.query_data('Bbox', condition2))
    # update data
    updated_data = {'frame_number': 2}
    ogm.update_data('Frame', updated_data, condition)

    # query data, check if update successfully
    print(ogm.query_data('Frame', condition))
    print(ogm.query_data('Bbox', condition2))
    
    
    cam_id = "video_1"
    time_start = 1
    print(ogm.query_car_id_count(cam_id, time_start))
    print(ogm.query_most_car_frame(cam_id, time_start))
    
    # Delete data
    ogm.delete_data('Frame', condition)
    ogm.delete_data('Bbox', condition2)

    # check if delete successfully
    print(ogm.query_data('Frame', condition))
    print(ogm.query_data('Bbox', condition2))

    # close connection
    ogm.close()
    
def parse_xml(xml_path):
    #Read the xml file, parse the xml file, 
    # and get the root node information
    if not os.path.exists(xml_path):
        return None
    rf = open(xml_path,"r")
    tree = ET.parse(rf)
    root = tree.getroot()
    frame = int(str(root.find("filename").text.split('.')[0]))
    cam_id = str(root.find("folder").text.split('_')[1])
    # print(frame,cam_id)
    # Analyze the box information of the picture
    # Traverse the information of the object node
    label_infos = []
    for object in root.iter("object"):
        # car id
        name = object.find("name").text
        # The model name of the vehicle
        car_type = object.find("pose").text
        # bbox
        bndbox = object.find("bndbox")
        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)
        box = (xmin,ymin,xmax,ymax)
        label_infos.append((name,car_type,box))
    return label_infos

def xml_suffix(frame):
    if frame < 10:
        suffix = "00000" + str(frame)
    elif frame >= 10 and frame < 100:
        suffix = "0000" + str(frame)
    elif frame >= 100 and frame < 1000:
        suffix = "000" + str(frame)
    elif frame >= 1000 and frame < 10000:
        suffix = "00" + str(frame)
    elif frame >= 10000 and frame < 100000:
        suffix = "0" + str(frame)
    return suffix


def insert_many_data(ogm,cam_id,frame_start,frame_end):
    traj_path = '/data/fzx/city/bbox'
    cam_bbox_dir =os.listdir(os.path.join(traj_path,'camera_'+str(cam_id)))

    for frame_id in range(frame_start,frame_end):        
        car_label = parse_xml(str(os.path.join(traj_path,'camera_'+str(cam_id),xml_suffix(frame_id)+'.xml')))

        frame = frame_id
        if car_label is None:
            continue
        for car in car_label:
            
            bbox_id = str(cam_id)+'_'+str(frame)+'_'+str(car[0])
            object_id = str(car[0])
            xmin,ymin,xmax,ymax = car[2][0],car[2][1],car[2][2],car[2][3]
            # insert data into Frame table
            frame_data = {
                'video_id': cam_id,
                'frame_number': frame,
                'bbox_id': bbox_id
            }
            ogm.insert_data('Frame', frame_data)
            # insert data into Bbox table
            bbox_data = {
                'bbox_id': bbox_id,
                'object_id': object_id,
                'class': 'car',
                'bbox_xmin': xmin,
                'bbox_ymin': ymin,
                'bbox_xmax': xmax,
                'bbox_ymax': ymax
            }
            ogm.insert_data('Bbox',bbox_data)

    # two special SQL queries
    cam_id = 4
    time_start = 1
    print(ogm.query_car_id_count(cam_id, time_start))
    print(ogm.query_most_car_frame(cam_id, time_start))
def main():
    # test_opengauss_manager()
    
        # init OpenGaussManager class
    ogm = OpenGaussManager(dbname="test_db", user="gaussdb", password="Secretpassword@123", host="127.0.0.1", port="15432")
    #  delete all tables
    ogm.cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public';")
    tables = ogm.cur.fetchall()
    for table in tables:
        # delete each table
        ogm.cur.execute(sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE;").format(sql.Identifier(table[0])))

    # create_tables()
    ogm.create_tables()
    # innser the cam4 data(from frame 1 to frame 100)
    insert_many_data(ogm,4,1,100)
    # TODO add more test cases
    # e.g. Test a whole video xml files

if __name__ == "__main__":
    main()
