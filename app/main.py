from fastapi import FastAPI, HTTPException
from .models import GaussSeidelRequest, GaussSeidelResponse
from .services.gauss_seidel import gauss_seidel
from .utils.matrix_utils import MatrixValidator, convergence_check
import numpy as np

app = FastAPI(title="API del Método de Gauss-Seidel")

@app.post("/solve", response_model=GaussSeidelResponse)
async def solve(request: GaussSeidelRequest):
    try:
        # Validar la matriz y vectores de entrada
        validator = MatrixValidator(request.A, request.b, request.x0)

        # Verificar criterios de convergencia
        convergence_info = validator.check_gauss_seidel_convergence()

        if not convergence_info["will_converge"]:
            warnings = "; ".join(convergence_info["warnings"])
            raise HTTPException(status_code=400, detail=f"El método puede no converger: {warnings}")

        # Resolver usando el método de Gauss-Seidel
        solution, iterations, error, converged = gauss_seidel(
            request.A,
            request.b,
            request.x0,
            request.max_iterations,
            request.tolerance
        )

        return GaussSeidelResponse(
            solution=solution,
            iterations=iterations,
            error=error,
            converged=converged
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_matrix(request: GaussSeidelRequest):
    """Analiza la matriz para determinar si el método de Gauss-Seidel convergerá"""
    try:
        # Validar la matriz y vectores de entrada
        validator = MatrixValidator(request.A, request.b, request.x0)

        # Obtener resumen de validación
        validation_summary = validator.get_validation_summary()

        # Estimar número de iteraciones
        estimated_iterations = validator.estimate_iterations(request.tolerance)

        return {
            "validation_summary": validation_summary,
            "estimated_iterations": estimated_iterations,
            "tolerance": request.tolerance
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}