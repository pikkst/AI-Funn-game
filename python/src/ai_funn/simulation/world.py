from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_funn.experiments.config import (
    CANONICAL_MAX_EPISODE_STEPS,
    CANONICAL_WORLD_HEIGHT,
    CANONICAL_WORLD_WIDTH,
    ConfigurationError,
    EnvironmentConfig,
)

from .resources import consume_resource, regrow_resources
from .rng import WorldRNG
from .spawn import find_valid_spawn, viable_surroundings
from .terrain import Terrain, is_passable

MAX_GENERATION_ATTEMPTS = 100


@dataclass(frozen=True)
class WorldState:
    width: int
    height: int
    max_steps: int
    terrain: tuple[tuple[int, ...], ...]
    food: tuple[tuple[float, ...], ...]
    organism_x: int
    organism_y: int
    energy: float
    health: float
    hunger: float
    step: int
    events: tuple[str, ...]
    rng_state: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "max_steps": self.max_steps,
            "terrain": [list(row) for row in self.terrain],
            "food": [list(row) for row in self.food],
            "organism_x": self.organism_x,
            "organism_y": self.organism_y,
            "energy": self.energy,
            "health": self.health,
            "hunger": self.hunger,
            "step": self.step,
            "events": list(self.events),
            "rng_state": _serialize_rng_state(self.rng_state),
        }

    @property
    def rng(self) -> WorldRNG:
        return WorldRNG.from_state(self.rng_state)


def _serialize_rng_state(state: Any) -> Any:
    # random.Random.getstate() returns (version, tuple, gauss_next)
    version, tuple_state, gauss_next = state
    return {"version": version, "state": list(tuple_state), "gauss_next": gauss_next}


def _deserialize_rng_state(data: Any) -> Any:
    if isinstance(data, dict):
        return (data["version"], tuple(data["state"]), data["gauss_next"])
    return data


def world_state_from_dict(data: dict[str, Any]) -> WorldState:
    if not isinstance(data, dict):
        raise ConfigurationError("world state must be dict", field_path=None)
    rng_state = _deserialize_rng_state(data["rng_state"])
    return WorldState(
        width=int(data["width"]),
        height=int(data["height"]),
        max_steps=int(data["max_steps"]),
        terrain=tuple(tuple(int(v) for v in row) for row in data["terrain"]),
        food=tuple(tuple(float(v) for v in row) for row in data["food"]),
        organism_x=int(data["organism_x"]),
        organism_y=int(data["organism_y"]),
        energy=float(data["energy"]),
        health=float(data["health"]),
        hunger=float(data["hunger"]),
        step=int(data["step"]),
        events=tuple(str(v) for v in data["events"]),
        rng_state=rng_state,
    )


WorldState.from_dict = staticmethod(world_state_from_dict)  # type: ignore[attr-defined]


def validate_world_state(state: WorldState) -> None:
    if state.width != CANONICAL_WORLD_WIDTH or state.height != CANONICAL_WORLD_HEIGHT:
        raise ConfigurationError("invalid world dimensions", field_path="world")
    if state.max_steps != CANONICAL_MAX_EPISODE_STEPS:
        raise ConfigurationError("invalid max steps", field_path="world")
    if len(state.terrain) != state.height or any(len(r) != state.width for r in state.terrain):
        raise ConfigurationError("terrain shape mismatch", field_path="terrain")
    if len(state.food) != state.height or any(len(r) != state.width for r in state.food):
        raise ConfigurationError("food shape mismatch", field_path="food")
    if not (0 <= state.organism_x < state.width and 0 <= state.organism_y < state.height):
        raise ConfigurationError("organism out of bounds", field_path="organism")
    if not is_passable(Terrain(state.terrain[state.organism_y][state.organism_x])):
        raise ConfigurationError("organism on impassable tile", field_path="organism")
    if not viable_surroundings(
        [list(r) for r in state.terrain], state.organism_x, state.organism_y
    ):
        raise ConfigurationError("organism lacks viable surroundings", field_path="organism")
    for row in state.food:
        for v in row:
            if v < 0 or v != v:  # nan check
                raise ConfigurationError("food resource invalid", field_path="food")
    if state.energy < 0 or state.health < 0 or state.hunger < 0:
        raise ConfigurationError("organism stats invalid", field_path="organism")
    if state.step > state.max_steps:
        raise ConfigurationError("step exceeds max", field_path="step")


def _generate_terrain(rng: WorldRNG, width: int, height: int) -> list[list[int]]:
    grid: list[list[int]] = [[int(Terrain.GROUND) for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            r = rng.random()
            if r < 0.12:
                grid[y][x] = int(Terrain.WATER)
            elif r < 0.32:
                grid[y][x] = int(Terrain.FOREST)
            elif r < 0.52:
                grid[y][x] = int(Terrain.FOOD)
            else:
                grid[y][x] = int(Terrain.GROUND)
    return grid


def generate_world(seed: int, environment_config: EnvironmentConfig) -> WorldState:
    if not isinstance(environment_config, EnvironmentConfig):
        raise ConfigurationError(
            "environment_config must be EnvironmentConfig", field_path="config"
        )
    width = environment_config.world_width
    height = environment_config.world_height
    max_steps = environment_config.max_episode_steps
    if width != CANONICAL_WORLD_WIDTH or height != CANONICAL_WORLD_HEIGHT:
        raise ConfigurationError("invalid world dimensions", field_path="environment")
    if max_steps != CANONICAL_MAX_EPISODE_STEPS:
        raise ConfigurationError("invalid max steps", field_path="environment")
    if environment_config.environment_version != "survival-v1":
        raise ConfigurationError("invalid environment version", field_path="environment")

    rng = WorldRNG(seed)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        terrain = _generate_terrain(rng, width, height)
        food = init_food_grid_from_terrain(terrain)
        try:
            sx, sy = find_valid_spawn(rng, terrain)
        except ValueError:
            continue
        state = WorldState(
            width=width,
            height=height,
            max_steps=max_steps,
            terrain=tuple(tuple(row) for row in terrain),
            food=tuple(tuple(row) for row in food),
            organism_x=sx,
            organism_y=sy,
            energy=100.0,
            health=100.0,
            hunger=0.0,
            step=0,
            events=(),
            rng_state=rng.get_state(),
        )
        try:
            validate_world_state(state)
        except ConfigurationError:
            continue
        return state
    raise ConfigurationError("failed to generate valid world", field_path="world")


def init_food_grid_from_terrain(terrain: list[list[int]]) -> list[list[float]]:
    from .resources import init_food_grid

    return init_food_grid(terrain, 1.0)


# attach to_dict/from_dict already defined
WorldState.to_dict = WorldState.to_dict  # type: ignore


def step(state: WorldState, action: int) -> tuple[WorldState, float, bool, bool, dict[str, Any]]:
    if state.step >= state.max_steps:
        raise ConfigurationError("terminal state cannot be stepped", field_path="step")
    terrain = [list(r) for r in state.terrain]
    food = [list(r) for r in state.food]
    x, y = state.organism_x, state.organism_y
    rng = state.rng
    events: list[str] = []
    # 1 validate action/episode active done
    # 2 apply movement/eat/rest
    if action == 0:  # north
        ny = y - 1
        if 0 <= ny < state.height and is_passable(Terrain(terrain[ny][x])):
            y = ny
        else:
            events.append("invalid_move")
    elif action == 1:  # south
        ny = y + 1
        if 0 <= ny < state.height and is_passable(Terrain(terrain[ny][x])):
            y = ny
        else:
            events.append("invalid_move")
    elif action == 2:  # west
        nx = x - 1
        if 0 <= nx < state.width and is_passable(Terrain(terrain[y][nx])):
            x = nx
        else:
            events.append("invalid_move")
    elif action == 3:  # east
        nx = x + 1
        if 0 <= nx < state.width and is_passable(Terrain(terrain[y][nx])):
            x = nx
        else:
            events.append("invalid_move")
    elif action == 4:  # eat
        amount = consume_resource(food, x, y)
        if amount <= 0.0:
            events.append("failed_eat")
        else:
            events.append(f"ate:{amount}")
    elif action == 5:  # rest
        events.append("rest")
    else:
        raise ConfigurationError("invalid action", field_path="action")

    # 3 metabolic cost/hunger/env effects: none beyond storing
    # 4 regrowth/random events
    regrow_resources(terrain, food, rng)
    # 5 death/truncation
    new_step = state.step + 1
    energy = state.energy
    health = state.health
    hunger = state.hunger
    terminated = health <= 0.0
    truncated = (not terminated) and new_step >= state.max_steps
    # death on final step precedence
    if new_step >= state.max_steps and health <= 0.0:
        terminated = True
        truncated = False
    reward = -1.0 if terminated else 0.01
    new_state = WorldState(
        width=state.width,
        height=state.height,
        max_steps=state.max_steps,
        terrain=tuple(tuple(row) for row in terrain),
        food=tuple(tuple(row) for row in food),
        organism_x=x,
        organism_y=y,
        energy=energy,
        health=health,
        hunger=hunger,
        step=new_step,
        events=tuple(events),
        rng_state=rng.get_state(),
    )
    validate_world_state(new_state)
    info = {"events": events}
    return new_state, reward, terminated, truncated, info
