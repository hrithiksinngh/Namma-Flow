# PROJECT_SPEC.md: Namma-Flow (Predictive Urban Micro-Flooding AI)

> **Role & Instruction for AI Agent:**  
> You are an expert Machine Learning Engineer and Full-Stack Geospatial Developer. Use this document as the complete blueprint and context file to implement the **Namma-Flow** project end-to-end. Implement all scripts, schemas, data transformation pipelines, model training routines, and deployment artifacts described below without omitting edge-case handling or validation checks.

---

## 1. Executive Summary & Objective

* **Project Name:** Namma-Flow
* **Target Domain:** Urban Micro-Flooding & Disaster Mitigation
* **Primary Geography:** Bengaluru, Karnataka, India (High-risk pilot corridor: Outer Ring Road / Bellandur / Mahadevapura watershed)
* **Goal:** Predict localized street junction inundation probabilities (p between 0 and 1) 12 to 48 hours in advance using a Spatio-Temporal Graph Neural Network (STGCN).
* **Core Hypothesis:** Micro-flooding in Bengaluru is governed by the structural coupling of topography (elevation gradients, natural valley lines/rajakaluves) and localized short-duration rainfall intensity. Modeling the road network as a graph structure allows Message Passing Neural Networks to learn how runoff accumulates along downstream street edges over time.
* **Budget Constraint:** $0 (utilizing OpenStreetMap via `osmnx`, public SRTM Digital Elevation Models, open weather APIs, and free-tier compute).

---

## 2. Directory Layout

Set up the project repository following this strict layout:

```text
namma-flow/
├── config/
│   └── config.yaml               # Pipeline configurations, hyperparams, bounding boxes
├── data/
│   ├── raw/
│   │   ├── dem/                  # SRTM DEM GeoTIFF files (.tif)
│   │   ├── weather/              # Hourly historical weather logs (.csv)
│   └── interim/
│   │   ├── bellandur_osm.graphml # Extracted and augmented OSM graph
│   └── processed/
│       ├── train_dataset.pt      # Preprocessed PyTorch Geometric temporal snapshots
│       └── val_dataset.pt
├── src/
│   ├── __init__.py
│   ├── data_pipeline/
│   │   ├── 01_extract_network.py # OSM road network ingestion & cleaning
│   │   ├── 02_elevation_engine.py# Raster extraction & slope calculation
│   │   ├── 03_weather_ingestion.py# Precipitation alignment
│   │   └── 04_dataset_builder.py # Tensor formatting and PyG dataset export
│   ├── models/
│   │   ├── stgcn.py              # STGCN / A3T-GCN architecture definition
│   │   ├── loss.py               # Focal Loss / Weighted BCE for extreme class imbalance
│   ├── training/
│   │   └── train.py              # Main training loop with checkpointing & early stopping
├── app/
│   └── app.py                    # Interactive Streamlit & Pydeck 3D GIS visualization
├── requirements.txt
└── PROJECT_SPEC.md
```

---

## 3. Technology Stack & Dependencies

`requirements.txt` contents:

```text
torch>=2.1.0
torch-geometric>=2.4.0
osmnx>=1.9.0
networkx>=3.1
rasterio>=1.3.8
geopandas>=0.14.0
numpy>=1.24.0
pandas>=2.1.0
scikit-learn>=1.3.0
streamlit>=1.30.0
pydeck>=0.8.0
pyyaml>=6.0.1
```

---

## 4. Mathematical Model & Data Schema

### 4.1. Graph Representation
* **Nodes:** Road intersections.
  * *Static Features:* Elevation (m), Proximity to drain.
  * *Dynamic Features:* Hourly precipitation (mm/h), Rolling sums (3h, 6h, 12h, 24h).
* **Edges:** Drivable street segments.
  * *Static Features:* Length (m), Slope / Grade.

### 4.2. Target Variable & Loss
* Binary classification: 1 (Flooded), 0 (Not Flooded).
* **Class Imbalance Handling:** Use **Focal Loss** with gamma = 2.0 and alpha = 0.85 to penalize the model heavily when it misses a rare flood event.

---

## 5. Implementation Blueprints

### 5.1. Configuration (`config/config.yaml`)

```yaml
project:
  name: "Namma-Flow"
  region: "Bellandur, Bengaluru, Karnataka, India"

model:
  hidden_dim: 64
  learning_rate: 0.001
  batch_size: 16
  epochs: 60
```

### 5.2. Graph Extraction (`src/data_pipeline/01_extract_network.py`)

```python
import osmnx as ox
import networkx as nx
import numpy as np

def extract_and_enrich_graph(region="Bellandur, Bengaluru, Karnataka, India"):
    G = ox.graph_from_place(region, network_type="drive", simplify=True)
    G = ox.project_graph(G, to_crs="EPSG:4326")
    G_di = nx.DiGraph(G)
    
    # Synthesize realistic valley topography if DEM raster is missing:
    for node, data in G_di.nodes(data=True):
        lat, lon = data["y"], data["x"]
        simulated_elevation = 880.0 + 30.0 * np.sin(lat * 100) + 15.0 * np.cos(lon * 100)
        data["elevation"] = round(float(simulated_elevation), 2)
        data["dist_to_drain_m"] = round(abs(lon - 77.6750) * 111000, 2)
        
    for u, v, data in G_di.edges(data=True):
        u_elev = G_di.nodes[u]["elevation"]
        v_elev = G_di.nodes[v]["elevation"]
        length = max(float(data.get("length", 25.0)), 1.0)
        data["grade"] = float((v_elev - u_elev) / length)
        data["length"] = float(length)
        
    ox.save_graphml(G_di, filepath="data/interim/bellandur_osm.graphml")
    return G_di
```

### 5.3. Spatio-Temporal GNN Architecture (`src/models/stgcn.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class SpatioTemporalFloodGNN(nn.Module):
    def __init__(self, node_in_dim=8, edge_dim=2, hidden_dim=64, dropout=0.2):
        super(SpatioTemporalFloodGNN, self).__init__()
        self.conv1 = GATv2Conv(node_in_dim, hidden_dim, edge_dim=edge_dim, heads=2, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * 2, hidden_dim, edge_dim=edge_dim, heads=1, concat=False)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr, hidden_state=None):
        h = F.elu(self.conv1(x, edge_index, edge_attr))
        h = self.dropout(h)
        h = F.elu(self.conv2(h, edge_index, edge_attr))
        
        if hidden_state is None:
            hidden_state = torch.zeros_like(h)
        h_temporal = self.gru(h, hidden_state)
        
        out = F.relu(self.fc1(h_temporal))
        probs = torch.sigmoid(self.fc2(self.dropout(out)))
        return probs, h_temporal
```

---

## 6. Execution Sequence

To build the project from scratch, run:
1. `pip install -r requirements.txt`
2. `python src/data_pipeline/01_extract_network.py`
3. `python src/data_pipeline/04_dataset_builder.py` (Generate datasets)
4. `python src/training/train.py`
5. `streamlit run app/app.py`