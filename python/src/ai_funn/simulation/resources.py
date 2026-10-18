from __future__ import annotations

from typing import Any

from .terrain import Terrain


def init_food_grid(terrain: list[list[int]], food_amount: float) -> list[list[float]]:
    height = len(terrain)
    width = len(terrain[0]) if height else 0
    grid: list[list[float]] = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if terrain[y][x] == int(Terrain.FOOD):
                grid[y][x] = float(food_amount)
    return grid


def consume_resource(food_grid: list[list[float]], x: int, y: int) -> float:
    if y < 0 or y >= len(food_grid):
        return 0.0
    row = food_grid[y]
    if x < 0 or x >= len(row):
        return 0.0
    amount = row[x]
    if amount > 0.0:
        row[x] = 0.0
        return float(amount)
    return 0.0


def regrow_resources(
    terrain: list[list[int]],
    food_grid: list[list[float]],
    rng: Any,
    regrow_probability: float = 0.02,
    regrow_amount: float = 3.0,
    capacity: float = 3.0,
) -> None:
    height = len(terrain)
    width = len(terrain[0]) if height else 0
    for y in range(height):
        for x in range(width):
            if terrain[y][x] == int(Terrain.FOOD):
                if food_grid[y][x] <= 0.0 and rng.random() < regrow_probability:
                    food_grid[y][x] = min(capacity, regrow_amount)
                if food_grid[y][x] > capacity:
                    food_grid[y][x] = capacity
