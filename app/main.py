from fastapi import FastAPI, HTTPException
from .models import GaussSeidelRequest, GaussSeidelResponse, IterationData
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
        warnings = convergence_info.get("warnings", [])

        # Obtener comparación con Jacobi
        comparison_with_jacobi = validator.compare_with_jacobi()

        # Resolver usando el método de Gauss-Seidel
        solution, iterations, error, converged, iteration_history = gauss_seidel(
            request.A,
            request.b,
            request.x0,
            request.max_iterations,
            request.tolerance
        )

        # Convertir el historial de iteraciones al formato adecuado
        formatted_history = [
            IterationData(
                iteration=item["iteration"],
                solution=item["solution"],
                error=item["error"]
            ) for item in iteration_history
        ]

        return GaussSeidelResponse(
            solution=solution,
            iterations=iterations,
            error=error,
            converged=converged,
            iteration_history=formatted_history,
            warnings=warnings,
            convergence_details=convergence_info.get("details", {}),
            comparison_with_jacobi=comparison_with_jacobi
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