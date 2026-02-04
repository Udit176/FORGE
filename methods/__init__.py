"""Learning methods module."""
from .policy import MLPPolicy
from .dagger import DAgger
from .mindmeld import MindMeld
from .cognitive_model import CognitiveModel

__all__ = ['MLPPolicy', 'DAgger', 'MindMeld', 'CognitiveModel']
