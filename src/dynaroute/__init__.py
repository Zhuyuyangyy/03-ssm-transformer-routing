"""DynaRoute: Dynamic SSM-Transformer Routing for Efficient Sequence Modeling.

A hybrid architecture that dynamically routes tokens between SSM (State Space
Model) and Transformer attention pathways for efficient sequence modeling with
adaptive computation.
"""

from .hybrid_layer import HybridLayer, SSMPathway, AttentionPathway, DynaRouteBlock
from .router import TokenRouter, AdaptiveRouter
from .training import ProgressiveTrainer, TrainingConfig, TemperatureScheduler
from .analysis import RoutingAnalyzer

__version__ = "0.2.0"
__all__ = [
    # Core layers
    "HybridLayer",
    "SSMPathway",
    "AttentionPathway",
    "DynaRouteBlock",
    # Routers
    "TokenRouter",
    "AdaptiveRouter",
    # Training
    "ProgressiveTrainer",
    "TrainingConfig",
    "TemperatureScheduler",
    # Analysis
    "RoutingAnalyzer",
]
