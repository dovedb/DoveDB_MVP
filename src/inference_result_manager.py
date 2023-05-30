import time
import json

# 
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.common import *
from typing import Dict
import pandas as pd
import prettytable
from nebula3.data.DataObject import Value, ValueWrapper
from nebula3.data.ResultSet import ResultSet

# print response in table format 
cast_as = {
    Value.NVAL: "as_null",
    Value.__EMPTY__: "as_empty",
    Value.BVAL: "as_bool",
    Value.IVAL: "as_int",
    Value.FVAL: "as_double",
    Value.SVAL: "as_string",
    Value.LVAL: "as_list",
    Value.UVAL: "as_set",
    Value.MVAL: "as_map",
    Value.TVAL: "as_time",
    Value.DVAL: "as_date",
    Value.DTVAL: "as_datetime",
    Value.VVAL: "as_node",
    Value.EVAL: "as_relationship",
    Value.PVAL: "as_path",
    Value.GGVAL: "as_geography",
    Value.DUVAL: "as_duration",
}
def customized_cast_with_dict(val: ValueWrapper):
    _type = val._value.getType()
    method = cast_as.get(_type)
    if method is not None:
        return getattr(val, method, lambda *args, **kwargs: None)()
    raise KeyError("No such key: {}".format(_type))
def print_resp(resp: ResultSet):
    assert resp.is_succeeded()
    output_table = prettytable.PrettyTable()
    output_table.field_names = resp.keys()
    for recode in resp:
        value_list = []
        for col in recode:
            val = customized_cast_with_dict(col)
            value_list.append(val)
        output_table.add_row(value_list)
    print(output_table)


class InferenceResultManager(object):
    """Inference result manager."""

    def __init__(self):
        # query result list
        self.eventid = None
        
        # connect nebula graphdb 
        config = Config()
        config.max_connection_pool_size = 10
        # init connection pool
        connection_pool = ConnectionPool()
        assert connection_pool.init([('127.0.0.1', 9669)], config)

        # get session from the pool
        self.client = connection_pool.get_session('root', 'nebula')
        assert self.client is not None

        # get the result in json format
        resp_json = self.client.execute_json("yield 1")
        json_obj = json.loads(resp_json)

    def create_schema(self):
        reps = self.client.execute(
            'CREATE SPACE  IF NOT EXISTS video_space(vid_type=FIXED_STRING(12));USE video_space;'
            'CREATE TAG IF NOT EXISTS Frame(video_id string, frame_number int);'
            'CREATE TAG Object(object_id string, class string);'
            'CREATE TAG Bbox(bbox_id string, bbox_xmin double, bbox_ymin double, bbox_xmax double, bbox_ymax double);'
            'CREATE EDGE frame_contains_object();'
            'CREATE EDGE frame_contains_bbox();'
        )
        assert reps.is_succeeded(), reps.error_msg()
        # insert data need to sleep after create schema
        time.sleep(12)
        return reps.is_succeeded()
    
    def insert_video_metadata(self,frame_data):
        query = """
        INSERT VERTEX Frame(video_id, frame_number)
        VALUES "{video_id}":("{video_id}", {frame_number})
        """
        for frame in frame_data:
            video_id = frame['video_id']
            frame_number = frame['frame_number']
            insert_query = query.format(video_id=video_id, frame_number=frame_number)
            print(insert_query) 
            reps = self.client.execute(insert_query)
            assert reps.is_succeeded(), reps.error_msg()
        return reps.is_succeeded()
            
    def insert_video_object(self,object_data):
        query = """
        INSERT VERTEX Object(object_id, class)
        VALUES "{object_id}":("{object_id}", "{class_name}")
        """
        for obj in object_data:
            object_id = obj['object_id']
            class_name = obj['class']
            insert_query = query.format(object_id=object_id, class_name=class_name)
            print(insert_query)
            reps = self.client.execute(insert_query)
            assert reps.is_succeeded(), reps.error_msg()
        return reps.is_succeeded()
    
    def insert_bbox(self,object_data):
        query = """
        INSERT VERTEX Bbox(bbox_id, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        VALUES "{bbox_id}":("{bbox_id}", {bbox_xmin}, {bbox_ymin}, {bbox_xmax}, {bbox_ymax})
        """
        for obj in object_data:
            bbox_id = obj['bbox_id']
            bbox_xmin,bbox_ymin = obj['bbox_xmin'],obj['bbox_ymin']
            bbox_xmax,bbox_ymax = obj['bbox_xmax'],obj['bbox_ymax']
            insert_query = query.format(bbox_id=bbox_id, bbox_xmin=bbox_xmin,
                                        bbox_ymin=bbox_ymin,bbox_xmax=bbox_xmax,bbox_ymax=bbox_ymax)
            print(insert_query)
            reps = self.client.execute(insert_query)
            assert reps.is_succeeded(), reps.error_msg()
        return reps.is_succeeded()
    
    def insert_bbox_frames_obj_edges(self,frame_conatin_bbox_data):
        
        query = """
            INSERT EDGE frame_contains_object()
            VALUES "{video_id},{frame_number}"->"{bbox_id}":()
        """
        for frame_conatin_bbox in frame_conatin_bbox_data:
            video_id = frame_conatin_bbox['video_id']
            frame_number = frame_conatin_bbox['frame_number']
            bbox_id = frame_conatin_bbox['bbox_id']
            insert_query = query.format(
                video_id=video_id, frame_number=frame_number
                ,bbox_id=bbox_id)
            
            print(insert_query)
            reps = self.client.execute(insert_query)
            assert reps.is_succeeded(), reps.error_msg()
        return reps.is_succeeded()
    
    def insert_object_contain_bbox(self,object_contain_bbox):
        query = """
            INSERT EDGE frame_contains_bbox()
            VALUES "{object_id}"->"{bbox_id}":()
        """
        for frame_conatin_bbox in object_contain_bbox:
            object_id = frame_conatin_bbox['object_id']
            bbox_id = frame_conatin_bbox['bbox_id']
            insert_query = query.format(
                object_id=object_id,bbox_id=bbox_id)
            
            print(insert_query)
            reps = self.client.execute(insert_query)
            assert reps.is_succeeded(), reps.error_msg()
        return reps.is_succeeded()
     
    def query_node(self,node_id):
        query = """          
            FETCH PROP ON Object "{node_id}" YIELD vertex as node
        """
        
        query_node = query.format(
        node_id=node_id)
        # get node property
        resp = self.client.execute(query_node)
        assert resp.is_succeeded(), resp.error_msg()
        print_resp(resp)
        return resp.is_succeeded()
    
    def query_edge(self,cam_id,video_id,bbox_id):
        query = f"""          
            FETCH PROP ON frame_contains_object "{cam_id},{video_id}"->"{bbox_id}" YIELD edge as e
        """
        query_edge = query.format(
        cam_id=cam_id,video_id=video_id,bbox_id=bbox_id)
        # get edge property
        resp = self.client.execute(query_edge)
        assert resp.is_succeeded(), resp.error_msg()
        print_resp(resp)
        return resp.is_succeeded()

    def drop_database(self):
        """Drop the database."""
        reps = self.client.execute('DROP SPACE video_space')
        assert reps.is_succeeded(), reps.error_msg()
        return reps.is_succeeded()

        
if __name__ == '__main__':
    client = None
    result_manager = InferenceResultManager()
    try:

        # create schema
        result_manager.create_schema()
        # insert vertex
        # example video frames data
        video_frames = [
            {
                'video_id': "cam0",
                'frame_number': 1,
            },
            {
                'video_id': "cam1",
                'frame_number': 2,
            }
            # ... other video frames data
        ]
        # bbox data, how to correspond to the frame data
        object_data = [
            {
                'object_id': 'obj0',
                'class': 'redcar',
            },
            {
                'object_id': 'obj1',
                'class': 'greencar',
            }]
        
        bbox_data = [
            {
                'bbox_id': 'bbox0', # need to generate unique id
                'bbox_xmin': 0.1,
                'bbox_ymin': 0.1,
                'bbox_xmax': 0.2,
                'bbox_ymax': 0.2
            },
            {
                'bbox_id': 'bbox1',
                'bbox_xmin': 0.5,
                'bbox_ymin': 0.1,
                'bbox_xmax': 0.2,
                'bbox_ymax': 0.2
            }]
        
            # example frame contains object data
        frame_contain_obj = [
            {
                'video_id': 'cam0',
                'frame_number': 3,
                'bbox_id': 'bbox0'
            },
            # ... other frame contains object data
        ]
        
        object_contain_bbox = [
            {
                'object_id': "obj0",
                'bbox_id': 'bbox0'
            },
            # ... other frame contains object data
        ]

        # import video metadata
        result_manager.insert_video_metadata(video_frames)
        # import bbox data
        result_manager.insert_video_object(object_data)
        # import bbox data
        result_manager.insert_bbox(bbox_data)
        # import edge data
        result_manager.insert_bbox_frames_obj_edges(frame_contain_obj)   
        # import edge data bbox object contains bbox
        result_manager.insert_object_contain_bbox(object_contain_bbox)
        
        # query node
        result_manager.query_node("obj0")
        # query edge
        result_manager.query_edge("cam0","3","bbox0")
        # drop database
        result_manager.drop_database()
        
    except Exception as x:
        import traceback

        print(traceback.format_exc())
        if client is not None:
            client.release()
        exit(1)
