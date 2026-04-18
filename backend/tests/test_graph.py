from backend.core.geometry import Zone, Opening
from backend.core.graph_model import SpatialGraph

# Create fake zones
z1 = Zone("Z1", [(0,0), (0,5), (5,5), (5,0)])
z2 = Zone("Z2", [(5,0), (5,5), (10,5), (10,0)])
z3 = Zone("Z3", [(10,0), (10,5), (15,5), (15,0)])

# Create graph
graph = SpatialGraph()

graph.add_zone(z1)
graph.add_zone(z2)
graph.add_zone(z3)

# Add connections
o1 = Opening("O1", "Z1", "Z2", width=1.5)
o2 = Opening("O2", "Z2", "Z3", width=1.0)

graph.add_opening(o1)
graph.add_opening(o2)

# Test path
print("Path Z1 → Z3:", graph.shortest_path("Z1", "Z3"))

# Test connectivity
print("Connected:", graph.is_fully_connected())