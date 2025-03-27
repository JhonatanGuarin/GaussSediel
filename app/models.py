from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class GaussSeidelRequest(BaseModel):
    A: List[List[float]]
    b: List[float]
    x0: Optional[List[float]] = None
    max_iterations: int = 100
    tolerance: float = 1e-10

class IterationData(BaseModel):
    iteration: int
    solution: List[float]
    error: float

class GaussSeidelResponse(BaseModel):
    solution: List[float]
    iterations: int
    error: float
    converged: bool
    iteration_history: Optional[List[IterationData]] = None
    warnings: Optional[List[str]] = None
    convergence_details: Optional[Dict[str, Any]] = None
    comparison_with_jacobi: Optional[Dict[str, Any]] = None