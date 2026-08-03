
"""Simple utilities for eigenvalue calculations.

Usage:
	from Matlab import eigenvalues
	vals, vecs = eigenvalues([[1,2],[3,4]])

Or run as a script to read a matrix from input.
"""
from typing import Tuple, Sequence
import sys

try:
	import numpy as _np
except Exception as _e:
	raise ImportError("NumPy is required for this module") from _e


def eigenvalues(matrix: Sequence[Sequence[float]]) -> Tuple[_np.ndarray, _np.ndarray]:
	"""Compute eigenvalues and eigenvectors of a square matrix.

	Args:
		matrix: square 2D sequence (list of lists or ndarray)

	Returns:
		(w, v) where w is 1D array of eigenvalues and v is 2D array of eigenvectors.
	"""
	arr = _np.array(matrix, dtype=float)
	if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
		raise ValueError("Input must be a square 2D matrix")
	w, v = _np.linalg.eig(arr)
	return w, v


def _main():
	"""Simple CLI: read whitespace-separated numbers per row; blank line to finish."""
	print("Enter matrix rows, numbers separated by spaces. Blank line to finish.")
	rows = []
	for line in sys.stdin:
		s = line.strip()
		if not s:
			break
		parts = s.split()
		rows.append([float(x) for x in parts])
	if not rows:
		print("No input provided")
		return
	try:
		w, v = eigenvalues(rows)
	except Exception as e:
		print(f"Error: {e}")
		return
	print("Eigenvalues:")
	for val in w:
		print(val)


if __name__ == '__main__':
	_main()
