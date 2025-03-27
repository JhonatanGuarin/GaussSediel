import numpy as np
from typing import List, Tuple, Optional, Dict, Any

def gauss_seidel(A: List[List[float]], b: List[float],
                x0: Optional[List[float]] = None,
                max_iterations: int = 100,
                tolerance: float = 1e-10) -> Tuple[List[float], int, float, bool, List[Dict[str, Any]]]:
    """
    Resuelve un sistema de ecuaciones lineales usando el método de Gauss-Seidel.

    Args:
        A: Matriz de coeficientes
        b: Vector del lado derecho
        x0: Aproximación inicial (por defecto: ceros)
        max_iterations: Número máximo de iteraciones
        tolerance: Tolerancia de error para convergencia

    Returns:
        Tupla que contiene:
        - solution: El vector solución
        - iterations: Número de iteraciones realizadas
        - error: Error final
        - converged: Si el método convergió
        - iteration_history: Historial de iteraciones
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Inicializa x0 si no se proporciona
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    # Contador de iteraciones
    iterations = 0
    error = float('inf')
    converged = False

    # Historial de iteraciones
    iteration_history = []

    # Iteración de Gauss-Seidel
    for iteration in range(max_iterations):
        x_old = x.copy()

        for i in range(n):
            # Calcula la suma para j < i (valores ya actualizados)
            sum1 = sum(A[i][j] * x[j] for j in range(i))

            # Calcula la suma para j > i (valores antiguos)
            sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))

            # Actualiza x[i]
            x[i] = (b[i] - sum1 - sum2) / A[i][i]

        # Calcula el error
        error = np.linalg.norm(x - x_old, np.inf)
        iterations += 1

        # Guardar el estado de esta iteración
        iteration_history.append({
            "iteration": iteration + 1,
            "solution": x.tolist(),
            "error": float(error)
        })

        # Verifica la convergencia
        if error < tolerance:
            converged = True
            break

    return x.tolist(), iterations, float(error), converged, iteration_history