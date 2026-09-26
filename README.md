## OHATP: Graph Anomaly Detection with Orthogonal-Hyperspherical Augmentation and Topology Perception
___
This is the source code for "OHATP: Graph Anomaly Detection with Orthogonal-Hyperspherical Augmentation and Topology Perception"


### Requirements
___
- python==3.7.16
- pytorch==1.12.0_cuda11.3
- dgl==0.4
- numpy==1.21.5


### Organization

The datasets containing injected anomalies are provided in the `dataset` folder. The expected directory layout is shown below:

```
.
├── ...
├── datasets
 │   ├── ACM.mat
 │   ├── Cora.mat
 │   ├── citeseer.mat
├── run.py
├── ...
```

   
### Running
___
    python run.py


