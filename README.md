# DoveDB

DoveDB 🕊️ is a database for intelligent video data management and analysis. Drawing inspiration from the world's most advanced vision AI methods, DoveDB represents a significant leap forward in open-source research and development in the field of video data handling.

## Introduction
Concerning the usability and efficiency to manage video data generated from large-scale cameras, we demonstrate DoveDB, a declarative and low-latency video database. We devise a more compre-
hensive video query language called VMQL to improve the expressiveness of previous SQL-like languages, which are augmented with functionalities for model-oriented management and deployment. We also propose a light-weight ingestion scheme to extract tracklets of all the moving objects and build semantic indexes to facilitate efficient query processing. For user interaction, we
construct a simulation environment with 120 cameras deployed in a road network and demonstrate three interesting scenarios. Using VMQL, users are allowed to 

1) train a visual model using SQL-like statement and deploy it on dozens of target cameras simultaneously for online inference; 

2) submit multi-object tracking (MOT) requests on target cameras, store the ingested results and build semantic indexes;

3) issue an aggregation or top-𝑘 query on the ingested cameras and obtain the response within milliseconds. A preliminary video introduction of DoveDB is available at

## Contribute
We greatly value your contributions and are committed to making your involvement with DoveDB as straightforward and transparent as possible. 

## License

**MIT License**: DoveDB is distributed under the MIT License, which is widely recognized for its permissive nature, allowing for various use cases, including commercial and open-source projects. It promotes flexibility, collaboration, and knowledge sharing. You can find more details in the LICENSE file in the DoveDB repository.
