from backend.core.geometry import Zone, Opening
from backend.core.graph_model import SpatialGraph
from backend.utils.validation import SystemValidator
from backend.utils.repair import AutoRepair

# create zones
z1 = Zone("Z1", [(0,0), (0,5), (5,5), (5,0)])
z2 = Zone("Z2", [(10,0), (10,5), (15,5), (15,0)])  # disconnected

graph = SpatialGraph()
graph.add_zone(z1)
graph.add_zone(z2)

fixes = AutoRepair.fix_disconnected_graph(graph)

print("Fixes applied:", fixes)
print("Now connected:", graph.is_fully_connected())