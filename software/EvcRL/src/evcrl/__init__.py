"""EvcRL: Gymnasium environment for camera-based EV eco-driving with sampled smooth execution.

Importing this package loads NumPy, Pillow and Gymnasium only (no Torch, no CUDA)
and does not modify any global configuration; every environment instance carries
its own route, rule profile and parameters.
"""
from gymnasium.envs.registration import register, registry
from ._version import __version__
from .env import EcoDriveEnv, normalized_to_command, command_to_normalized
from .memory import VisionMemory
from .execution import Executor
from .route import Route
from .oracle import OracleDetector
from .profiles import PROFILES, CONTRACT_VERSION, get_profile, profile_identity, source_identity

__all__ = ['EcoDriveEnv', 'VisionMemory', 'Executor', 'Route', 'OracleDetector', 'PROFILES', 'CONTRACT_VERSION',
           'get_profile', 'profile_identity', 'source_identity', 'normalized_to_command', 'command_to_normalized', '__version__']

for _name, _kwargs in [('EvcRL-Camera-v0', {'mode': 'camera', 'route_length': 4000.}),
                       ('EvcRL-Structured-v0', {'mode': 'structured', 'route_length': 20000.}),
                       ('EvcRL-StructuredMini-v0', {'mode': 'structured', 'route_length': 4000.})]:
    if _name not in registry:
        register(id=_name, entry_point='evcrl.env:EcoDriveEnv', kwargs=_kwargs)
