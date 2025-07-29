from typing import Optional
import numpy as np
import gymnasium as gym
from typing import Callable, Optional, Tuple
from gymnasium import spaces
from models.station import Station


class BiciMadEnv(gym.Env):
    """
    Entorno de simulación para el sistema BiciMad.
    Este entorno simula el movimiento de bicicletas entre estaciones de BiciMad.
    """
    def __init__(self, 
                 stations: list[Station], 
                 reward_fn: Callable[[np.ndarray, np.ndarray, Tuple[int, int, int]], float],
                 max_move: Optional[int] = 10, 
                 seed: Optional[int] = None, 
                 max_steps: Optional[int] = 100,  
        ):
        """Inicializa el entorno BiciMad."""
        super(BiciMadEnv, self).__init__()

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.stations = stations
        self.num_stations = len(stations)
        self.max_move = max_move
        self.max_steps = max_steps
        self.capacities = np.array([s.capacity for s in self.stations], dtype=np.int32)

        # Espacios
        # Observación: número de bicis por estación (0..capacidad)
        self.observation_space = spaces.MultiDiscrete(self.capacities + 1)

        # Acción: (origen, destino, cantidad)
        self.action_space = spaces.MultiDiscrete([self.num_stations, 
                                                  self.num_stations, self.max_move + 1])
        
        self.reward_fn = reward_fn

        # Estado interno
        self._initial_bikes = np.array([s.bikes for s in self.stations], dtype=np.int32)
        self.state = self._initial_bikes.copy()
        self._step_count = 0


    def _get_obs(self) -> np.ndarray:
        """Devuelve la observación actual del entorno."""
        return self.state.copy()
    
    def set_state(self, bikes: np.ndarray):
        """Actualiza las estaciones del entorno."""

        "El número de bicicletas debe coincidir con el número de estaciones."
        assert bikes.shape == (self.num_stations,)
        assert np.all(bikes >= 0) & np.all(bikes <= self.capacities)
        self.state = bikes.astype(np.int32)   


    def reset(self, seed: Optional[int] = None):
        """Reinicia el entorno y devuelve la observación inicial."""
        if seed is not None:
            self._rng = np.random.seed(seed)
        self.state = self._initial_bikes.copy()
        self._step_count = 0
        return self._get_obs(), {}
    

    def step(self, action : Tuple[int, int, int]):
        origin, destination, qty = map(int, action)
        prev_state = self.state.copy()

        # Validación de acción
        qty = min(qty, self.max_move)
        qty = min(qty, prev_state[origin])
        qty = min(qty, self.capacities[destination] - prev_state[destination])   

        # Actualizar el estado  
        self.state[origin] -= qty
        self.state[destination] += qty

        # Recompensa
        reward = self.reward_fn(prev_state, self.state, (origin, destination, qty))

        # Terminación / truncamiento
        self._step_count += 1
        terminated = False
        truncated = self.max_steps is not None and self._step_count >= self.max_steps

        info = {
            "moved": qty,
            "origin": origin,
            "dest": destination,
            "prev_state": prev_state,
        }
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self, mode: str = "human"):
        if mode != "human":
            raise NotImplementedError("Sólo 'human' está implementado.")
        # Print básico del estado de las estaciones
        for i, s in enumerate(self.stations):
            print(f"{s.name:25s} | {self.state[i]:2d}/{s.capacity:2d}")

    def render(self, mode='human'):
        pass

    def close(self):
        pass