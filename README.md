# Anomaly Detection
This repository contains the code for a custom anomaly detection algorithm designed to detect anomalies in the toy 2D dataset [anomaly.csv](anomaly.csv) and a simple REST API wrapping this algorithm.

## Files
- [anomaly.csv](anomaly.csv) : contains the dataset
- [anomaly.py](anomaly.py) : contains the script to perform the anomaly detection and the functions that are imported by [the REST API](rest_api.py)
- [rest_api.py](rest_api.py) : contains the basic REST API
- [anomaly_detection_report.pdf](anomaly_detection_report.pdf) : contains some more in depth information about the algorithm and the API as well as some [results](results) and [figures](figures) in report style.
- [figures](figures) : contains the image files used in [the report](anomaly_detection_report.pdf)
- [results](results) : contains the outliers the algorithm detected in [the dataset](anomaly.csv) in plain text

## More information
Read [the report](anomaly_detection_report.pdf) :simple_smile: