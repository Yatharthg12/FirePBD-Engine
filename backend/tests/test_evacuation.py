from backend.core.geometry import Zone, Opening
from backend.core.graph_model import SpatialGraph
from backend.core.grid_model import Grid
from backend.agents.fire_agent import FireSimulator
from backend.agents.evacuation_agent import EvacuationSimulator, Person

# zones
z1 = Zone("Z1", [(0,0),(0,5),(5,5),(5,0)])
z2 = Zone("Z2", [(5,0),(5,5),(10,5),(10,0)])
z3 = Zone("Z3", [(10,0),(10,5),(15,5),(15,0)])  # exit

zones = [z1, z2, z3]

graph = SpatialGraph()
for z in zones:
    graph.add_zone(z)

graph.add_opening(Opening("O1","Z1","Z2",1.5))
graph.add_opening(Opening("O2","Z2","Z3",1.5))

# fire
grid = Grid(100,100,5)
fire = FireSimulator(grid)
grid.ignite(20,20)

# evacuation
sim = EvacuationSimulator(
    graph=graph,
    fire_grid=grid,
    exit_zones=["Z3"]
)

person = Person("P1", "Z1")
sim.add_person(person)

for t in range(10):
    fire.step()
    sim.step()

    print(f"Step {t}: Zone={person.current_zone}, Evacuated={person.evacuated}")