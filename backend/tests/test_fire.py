from backend.core.grid_model import Grid
from backend.agents.fire_agent import FireSimulator

grid = Grid(100, 100, cell_size=5)

# ignite center
grid.ignite(50, 50)

sim = FireSimulator(grid)

for t in range(10):
    sim.step()
    print(f"Step {t}: Burning cells =", (grid.state == 1).sum())