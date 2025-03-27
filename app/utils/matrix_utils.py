import numpy as np
from typing import List, Dict, Any, Union
from fastapi import HTTPException

class MatrixValidator:
    def __init__(self, A: List[List[float]], b: List[float], initial_guess: List[float] = None):
        """
        Inicializa el validador de matrices para métodos iterativos.

        Args:
            A: Matriz de coeficientes
            b: Vector de términos independientes
            initial_guess: Vector inicial de aproximación (opcional)
        """
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.initial_guess = np.array(initial_guess, dtype=float) if initial_guess is not None else None
        self.n = len(b)

        # Realizar validaciones básicas
        self._validate_dimensions()
        self._validate_matrix_properties()
        if initial_guess is not None:
            self._validate_initial_guess()

    def _validate_dimensions(self):
        """
        Valida las dimensiones de la matriz A y los vectores b e initial_guess.
        """
        try:
            # Verificar si la matriz A está vacía
            if len(self.A) == 0:
                raise ValueError("La matriz de coeficientes A no puede estar vacía")

            # Verificar si el vector b está vacío
            if len(self.b) == 0:
                raise ValueError("El vector de términos independientes b no puede estar vacío")

            # Verificar si A es una matriz cuadrada
            if not all(len(row) == self.n for row in self.A):
                raise ValueError("La matriz A debe ser cuadrada (mismo número de filas y columnas)")

            # Verificar si las dimensiones de A y b son compatibles
            if len(self.A) != len(self.b):
                raise ValueError(f"La dimensión de la matriz A ({len(self.A)}x{len(self.A[0])}) no es compatible con el vector b (longitud {len(self.b)})")

        except ValueError as e:
            # Re-lanzar errores de valor con el mensaje original
            raise ValueError(str(e))
        except Exception as e:
            # Capturar cualquier otro error y proporcionar un mensaje claro
            raise ValueError(f"Error inesperado al validar dimensiones: {str(e)}")

    def _validate_matrix_properties(self):
        """
        Valida propiedades específicas de la matriz para métodos iterativos.
        """
        try:
            # Verificar si hay elementos diagonales nulos
            for i in range(self.n):
                if self.A[i, i] == 0:
                    raise ValueError(f"El elemento diagonal A[{i},{i}] es cero. Los métodos iterativos requieren elementos diagonales no nulos.")

            # Verificar si hay elementos NaN o infinitos en la matriz A
            if np.any(np.isnan(self.A)) or np.any(np.isinf(self.A)):
                raise ValueError("La matriz A contiene valores no válidos (NaN o infinito)")

            # Verificar si hay elementos NaN o infinitos en el vector b
            if np.any(np.isnan(self.b)) or np.any(np.isinf(self.b)):
                raise ValueError("El vector b contiene valores no válidos (NaN o infinito)")

        except ValueError as e:
            # Re-lanzar errores de valor con el mensaje original
            raise ValueError(str(e))
        except Exception as e:
            # Capturar cualquier otro error y proporcionar un mensaje claro
            raise ValueError(f"Error inesperado al validar propiedades de la matriz: {str(e)}")

    def _validate_initial_guess(self):
        """
        Valida el vector de aproximación inicial.
        """
        try:
            # Verificar si las dimensiones del vector inicial son compatibles
            if len(self.initial_guess) != self.n:
                raise ValueError(f"La dimensión del vector inicial ({len(self.initial_guess)}) no coincide con la dimensión del sistema ({self.n})")

            # Verificar si hay elementos NaN o infinitos en el vector inicial
            if np.any(np.isnan(self.initial_guess)) or np.any(np.isinf(self.initial_guess)):
                raise ValueError("El vector inicial contiene valores no válidos (NaN o infinito)")

        except ValueError as e:
            # Re-lanzar errores de valor con el mensaje original
            raise ValueError(str(e))
        except Exception as e:
            # Capturar cualquier otro error y proporcionar un mensaje claro
            raise ValueError(f"Error inesperado al validar el vector inicial: {str(e)}")

    def is_diagonally_dominant(self):
        """
        Verifica si la matriz A es diagonalmente dominante.

        Returns:
            bool: True si la matriz es diagonalmente dominante, False en caso contrario
        """
        for i in range(self.n):
            diagonal = abs(self.A[i, i])
            sum_others = sum(abs(self.A[i, j]) for j in range(self.n) if j != i)

            if diagonal <= sum_others:
                return False

        return True

    def is_symmetric_positive_definite(self):
        """
        Verifica si la matriz A es simétrica y definida positiva.

        Returns:
            bool: True si la matriz es simétrica y definida positiva, False en caso contrario
        """
        # Verificar si es simétrica
        if not np.allclose(self.A, self.A.T):
            return False

        # Verificar si es definida positiva (todos los valores propios son positivos)
        try:
            eigenvalues = np.linalg.eigvals(self.A)
            return np.all(eigenvalues > 0)
        except np.linalg.LinAlgError:
            return False

    def check_gauss_seidel_convergence(self) -> Dict[str, Any]:
        """
        Verifica las condiciones de convergencia para el método de Gauss-Seidel.

        Returns:
            Diccionario con información sobre las condiciones de convergencia
        """
        result = {
            "will_converge": True,
            "is_diagonally_dominant": False,
            "is_symmetric_positive_definite": False,
            "spectral_radius": None,
            "warnings": [],
            "details": {}
        }

        try:
            # Verificar dominancia diagonal
            result["is_diagonally_dominant"] = self.is_diagonally_dominant()

            if not result["is_diagonally_dominant"]:
                result["warnings"].append("La matriz no es diagonalmente dominante. La convergencia no está garantizada.")

            # Verificar si es simétrica y definida positiva
            result["is_symmetric_positive_definite"] = self.is_symmetric_positive_definite()

            if result["is_symmetric_positive_definite"]:
                result["details"]["spd_message"] = "La matriz es simétrica y definida positiva, lo que garantiza la convergencia del método de Gauss-Seidel."

            # Calcular el radio espectral de la matriz de iteración de Gauss-Seidel
            try:
                D = np.diag(np.diag(self.A))
                L = np.tril(self.A, -1)
                U = np.triu(self.A, 1)

                # Matriz de iteración de Gauss-Seidel: G = -(D+L)^(-1) * U
                DL_inv = np.linalg.inv(D + L)
                G = -np.dot(DL_inv, U)

                # Calcular los valores propios
                eigenvalues = np.linalg.eigvals(G)
                spectral_radius = max(abs(eigenvalues))

                result["spectral_radius"] = float(spectral_radius)
                result["details"]["eigenvalues"] = eigenvalues.tolist()

                if spectral_radius >= 1:
                    result["warnings"].append(f"El radio espectral de la matriz de iteración es {spectral_radius:.4f} >= 1. El método de Gauss-Seidel podría no converger.")
                    if not result["is_diagonally_dominant"] and not result["is_symmetric_positive_definite"]:
                        result["will_converge"] = False
                else:
                    result["details"]["estimated_iterations"] = int(np.ceil(np.log(1e-6) / np.log(spectral_radius)))
                    result["details"]["convergence_message"] = f"El radio espectral es {spectral_radius:.4f} < 1, lo que indica que el método convergerá."

            except np.linalg.LinAlgError:
                result["warnings"].append("No se pudo calcular el radio espectral. Posible problema numérico con la matriz.")

            # Verificar el condicionamiento de la matriz
            try:
                cond_number = np.linalg.cond(self.A)
                result["details"]["condition_number"] = float(cond_number)

                if cond_number > 1e6:
                    result["warnings"].append(f"La matriz está mal condicionada (número de condición: {cond_number:.2e}). Esto puede afectar la precisión de la solución.")
            except np.linalg.LinAlgError:
                result["warnings"].append("No se pudo calcular el número de condición. Posible problema numérico con la matriz.")

            # Verificar si Gauss-Seidel convergerá
            if not result["is_diagonally_dominant"] and not result["is_symmetric_positive_definite"]:
                if result["spectral_radius"] is None or result["spectral_radius"] >= 1:
                    result["will_converge"] = False
                    result["warnings"].append("La matriz no cumple ninguno de los criterios suficientes para garantizar la convergencia del método de Gauss-Seidel.")

            return result

        except Exception as e:
            # Capturar cualquier error y proporcionar un mensaje claro
            result["warnings"].append(f"Error al verificar condiciones de convergencia: {str(e)}")
            result["will_converge"] = False
            return result

    def compare_with_jacobi(self) -> Dict[str, Any]:
        """
        Compara la convergencia esperada con el método de Jacobi.

        Returns:
            Diccionario con información comparativa
        """
        try:
            # Calcular matriz de iteración de Jacobi
            D = np.diag(np.diag(self.A))
            D_inv = np.diag(1.0 / np.diag(self.A))
            R_jacobi = np.eye(self.n) - np.dot(D_inv, self.A)

            # Calcular matriz de iteración de Gauss-Seidel
            L = np.tril(self.A, -1)
            U = np.triu(self.A, 1)
            DL_inv = np.linalg.inv(D + L)
            R_gauss = -np.dot(DL_inv, U)

            # Calcular radios espectrales
            spec_radius_jacobi = max(abs(np.linalg.eigvals(R_jacobi)))
            spec_radius_gauss = max(abs(np.linalg.eigvals(R_gauss)))

            # Comparar
            comparison = {
                "jacobi_spectral_radius": float(spec_radius_jacobi),
                "gauss_seidel_spectral_radius": float(spec_radius_gauss),
                "gauss_seidel_faster": spec_radius_gauss < spec_radius_jacobi,
                "estimated_speedup": float(np.log(spec_radius_jacobi) / np.log(spec_radius_gauss)) if spec_radius_gauss < 1 and spec_radius_jacobi < 1 else None,
                "conclusion": ""
            }

            if spec_radius_gauss < spec_radius_jacobi:
                comparison["conclusion"] = f"El método de Gauss-Seidel convergerá aproximadamente {comparison['estimated_speedup']:.2f} veces más rápido que Jacobi."
            elif spec_radius_gauss == spec_radius_jacobi:
                comparison["conclusion"] = "Ambos métodos tienen tasas de convergencia similares."
            else:
                comparison["conclusion"] = "El método de Jacobi convergerá más rápido que Gauss-Seidel para esta matriz."

            return comparison

        except Exception as e:
            return {
                "error": f"No se pudo realizar la comparación: {str(e)}",
                "jacobi_spectral_radius": None,
                "gauss_seidel_spectral_radius": None,
                "gauss_seidel_faster": None
            }

    def estimate_iterations(self, tolerance: float = 1e-6) -> int:
        """
        Estima el número de iteraciones necesarias para alcanzar la tolerancia especificada.

        Args:
            tolerance: Tolerancia deseada

        Returns:
            Número estimado de iteraciones
        """
        try:
            # Calcular la matriz de iteración de Gauss-Seidel
            D = np.diag(np.diag(self.A))
            L = np.tril(self.A, -1)
            U = np.triu(self.A, 1)

            # Matriz de iteración de Gauss-Seidel: G = -(D+L)^(-1) * U
            DL_inv = np.linalg.inv(D + L)
            G = -np.dot(DL_inv, U)

            # Calcular el radio espectral
            eigenvalues = np.linalg.eigvals(G)
            spectral_radius = max(abs(eigenvalues))

            if spectral_radius >= 1:
                return float('inf')  # No convergerá

            # Estimar el número de iteraciones
            # Fórmula: log(tol) / log(spectral_radius)
            iterations = np.ceil(np.log(tolerance) / np.log(spectral_radius))
            return int(iterations)

        except Exception as e:
            # En caso de error, devolver un valor predeterminado
            return 100  # Valor predeterminado

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Proporciona un resumen de todas las validaciones.

        Returns:
            Diccionario con el resumen de validaciones
        """
        summary = {
            "dimensions": {
                "matrix_size": f"{self.n}x{self.n}",
                "vector_size": self.n,
                "is_square": True
            },
            "matrix_properties": {
                "has_zero_diagonal": False,
                "contains_invalid_values": False
            },
            "convergence": self.check_gauss_seidel_convergence(),
            "comparison_with_jacobi": self.compare_with_jacobi()
        }

        return summary

def is_diagonally_dominant(A):
    """Verifica si la matriz A es diagonalmente dominante"""
    A = np.array(A)
    n = A.shape[0]

    for i in range(n):
        diagonal = abs(A[i, i])
        sum_row = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diagonal <= sum_row:
            return False

    return True

def is_symmetric_positive_definite(A):
    """Verifica si la matriz A es simétrica y definida positiva"""
    A = np.array(A)

    # Verificar si es simétrica
    if not np.allclose(A, A.T):
        return False

    # Verificar si es definida positiva
    try:
        eigenvalues = np.linalg.eigvals(A)
        return np.all(eigenvalues > 0)
    except np.linalg.LinAlgError:
        return False

def convergence_check(A):
    """
    Verifica si el método de Gauss-Seidel convergerá para la matriz A.

    Args:
        A: Matriz de coeficientes

    Returns:
        Tuple: (convergerá, mensaje)
    """
    A = np.array(A)

    # Verificar si es diagonalmente dominante
    if is_diagonally_dominant(A):
        return True, "La matriz es diagonalmente dominante"

    # Verificar si es simétrica y definida positiva
    if is_symmetric_positive_definite(A):
        return True, "La matriz es simétrica y definida positiva"

    # Calcular el radio espectral de la matriz de iteración
    try:
        n = A.shape[0]
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)

        # Matriz de iteración de Gauss-Seidel: G = -(D+L)^(-1) * U
        DL_inv = np.linalg.inv(D + L)
        G = -np.dot(DL_inv, U)

        # Calcular el radio espectral
        spectral_radius = max(abs(np.linalg.eigvals(G)))

        if spectral_radius < 1:
            return True, f"El radio espectral de la matriz de iteración es {spectral_radius:.4f} < 1"
        else:
            return False, f"El radio espectral de la matriz de iteración es {spectral_radius:.4f} >= 1"
    except:
        return False, "No se pudo determinar la convergencia mediante el radio espectral"