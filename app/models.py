from pydantic import BaseModel
from typing import List, Optional

class GaussSeidelRequest(BaseModel):
    A: List[List[float]]
    b: List[float]
    x0: Optional[List[float]] = None
    max_iterations: int = 100
    tolerance: float = 1e-10

class GaussSeidelResponse(BaseModel):
    solution: List[float]
    iterations: int
    error: float
    converged: bool