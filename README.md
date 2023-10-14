# DoveDB

DoveDB 🕊️ is a database for intelligent video data management and analysis. Drawing inspiration from the world's most advanced vision AI methods, DoveDB represents a significant leap forward in open-source research and development in the field of video data handling.

## Introduction

DoveDB is a video database designed to handle and manage video data efficiently. This repository contains the MVP version of DoveDB.

## Requirements

1. Clone the repository
```bash
git clone https://github.com/dovedb/DoveDB_MVP.git
```

2. Ensure either `OpenGauss` or `PostgreSQL` is installed.

3. Install the necessary dependencies provided in the requirements.txt.
```bash
cd src && pip install -r requirements.txt 
```

## Quick start

1. Download the provided ETL (Extract, Transform, Load) data which will be used for demonstration purposes.

[Download ETL Data Here](https://drive.google.com/file/d/1KGcr2wEF9_9s_YRgSI30Uwk4QfPCCuyp/view?usp=sharing)

Once downloaded, move the file to the src/data directory.

2. Launch DoveDB Server
Navigate to the `src` directory and start the server:
```python
cd src
python run_server.py --load_etl_data data/parsed.json
```

3. Launch DoveDB Client
In another terminal instance, initiate the client:
```python
cd src
python client.py
```

4. Aggregation Query:

Count Cars in the First 120 Seconds:
```sql
SELECT COUNT(DISTINCT track_id) FROM cars WHERE timestamp < 120 AND confidence > 0.7;
```

Calculate Stay Duration of a Specific Car:
```sql
SELECT MAX(timestamp) - MIN(timestamp) as stay_duration FROM cars WHERE track_id = 20;
```

5. Selection Query:

Top Frames with Most Cars:
```sql
SELECT frameid, COUNT(track_id) as car_count FROM cars GROUP BY frameid ORDER BY car_count DESC LIMIT 5;
```


## Contribute
We greatly value your contributions and are committed to making your involvement with DoveDB as straightforward and transparent as possible. 

## License

**MIT License**: DoveDB is distributed under the MIT License, which is widely recognized for its permissive nature, allowing for various use cases, including commercial and open-source projects. It promotes flexibility, collaboration, and knowledge sharing. You can find more details in the LICENSE file in the DoveDB repository.
