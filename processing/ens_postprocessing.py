"""
backend/processing/ens_postprocessing.py

Ensemble post-processing utilities for NWP workflows.

Design goals:
- xarray-first API: all operations accept xarray DataArray/Dataset inputs
- dask-friendly: computations remain lazy until explicitly computed/written
- source-agnostic: callers control I/O and provide an in-memory or lazy Dataset
- practical diagnostics for convective and synoptic workflows

Typical workflow:
1) Load GRIB2/NetCDF to xarray Dataset (possibly chunked with dask)
2) Optionally convert/write to Zarr with retained grid metadata
3) Compute derived ensemble diagnostics as DataArrays
4) Attach diagnostics to an output Dataset and write to Zarr/NetCDF
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import xarray as xr


# -----------------------------------------------------------------------------
# Dimension / coordinate helpers
# -----------------------------------------------------------------------------

DEFAULT_MEMBER_CANDIDATES: tuple[str, ...] = (
	"member",
	"ensemble",
	"ens",
	"realization",
	"number",
)

DEFAULT_TIME_CANDIDATES: tuple[str, ...] = (
	"time",
	"valid_time",
)

DEFAULT_WINDOW_DIM_CANDIDATES: tuple[str, ...] = (
	"time",
	"valid_time",
)

DEFAULT_Y_CANDIDATES: tuple[str, ...] = ("y", "lat", "latitude", "nj")
DEFAULT_X_CANDIDATES: tuple[str, ...] = ("x", "lon", "longitude", "ni")


def _find_dim(da: xr.DataArray, candidates: Sequence[str]) -> str:
	for name in candidates:
		if name in da.dims:
			return name
	raise ValueError(
		f"Could not identify required dimension from candidates {tuple(candidates)}. "
		f"Available dims: {da.dims}"
	)


def _find_coord_name(ds: xr.Dataset, candidates: Sequence[str]) -> str | None:
	for name in candidates:
		if name in ds.coords:
			return name
		if name in ds.variables:
			return name
	return None


def _normalize_longitudes(values: np.ndarray) -> np.ndarray:
	# Preserve floats and convert to [-180, 180] range.
	norm = ((values + 180.0) % 360.0) - 180.0
	return norm.astype(np.float64, copy=False)


def _iter_xy_dims(da: xr.DataArray, y_dim: str | None = None, x_dim: str | None = None) -> tuple[str, str]:
	if y_dim is not None and x_dim is not None:
		return y_dim, x_dim

	resolved_y = y_dim
	if resolved_y is None:
		for cand in DEFAULT_Y_CANDIDATES:
			if cand in da.dims:
				resolved_y = cand
				break

	resolved_x = x_dim
	if resolved_x is None:
		for cand in DEFAULT_X_CANDIDATES:
			if cand in da.dims:
				resolved_x = cand
				break

	if resolved_y is not None and resolved_x is not None:
		return resolved_y, resolved_x

	# Fallback for unconventional naming: infer from trailing non-member dims
	non_member_dims = [d for d in da.dims if d not in DEFAULT_MEMBER_CANDIDATES]
	if len(non_member_dims) < 2:
		raise ValueError(
			"Could not infer spatial dimensions; provide y_dim/x_dim explicitly. "
			f"Available dims: {da.dims}"
		)

	inferred_y = non_member_dims[-2]
	inferred_x = non_member_dims[-1]

	return resolved_y or inferred_y, resolved_x or inferred_x


def _ordered_quantile(da: xr.DataArray, q: float, dim: str) -> xr.DataArray:
	# xarray handles dask-backed quantiles with lazy execution.
	out = da.quantile(q, dim=dim, skipna=True)
	if "quantile" in out.coords:
		out = out.squeeze("quantile", drop=True)
	return out


# -----------------------------------------------------------------------------
# Ensemble reduction utilities
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleDims:
	member_dim: str = "member"
	time_dim: str | None = "time"
	y_dim: str = "y"
	x_dim: str = "x"


def infer_ensemble_dims(
	da: xr.DataArray,
	member_dim: str | None = None,
	time_dim: str | None = None,
	y_dim: str | None = None,
	x_dim: str | None = None,
) -> EnsembleDims:
	"""
	Infer key dimensions for ensemble diagnostics.

	Parameters
	----------
	da:
		Input DataArray.
	member_dim, time_dim, y_dim, x_dim:
		Optional explicit overrides for dimension names.
	"""
	resolved_member = member_dim or _find_dim(da, DEFAULT_MEMBER_CANDIDATES)

	if time_dim is not None:
		if time_dim not in da.dims:
			raise ValueError(f"time_dim '{time_dim}' not found in dims {da.dims}")
		resolved_time: str | None = time_dim
	else:
		resolved_time = next((d for d in DEFAULT_TIME_CANDIDATES if d in da.dims), None)

	resolved_y, resolved_x = _iter_xy_dims(da, y_dim=y_dim, x_dim=x_dim)
	return EnsembleDims(
		member_dim=resolved_member,
		time_dim=resolved_time,
		y_dim=resolved_y,
		x_dim=resolved_x,
	)


def mean_field(da: xr.DataArray, member_dim: str = "member") -> xr.DataArray:
	"""Ensemble mean over the member dimension."""
	return da.mean(dim=member_dim, skipna=True)


def spread_field(da: xr.DataArray, member_dim: str = "member") -> xr.DataArray:
	"""Ensemble spread (standard deviation) over the member dimension."""
	return da.std(dim=member_dim, skipna=True)


def max_field(da: xr.DataArray, member_dim: str = "member") -> xr.DataArray:
	"""Ensemble maximum over member dimension."""
	return da.max(dim=member_dim, skipna=True)


def min_field(da: xr.DataArray, member_dim: str = "member") -> xr.DataArray:
	"""Ensemble minimum over member dimension."""
	return da.min(dim=member_dim, skipna=True)


def probability_exceedance(
	da: xr.DataArray,
	threshold: float,
	member_dim: str = "member",
	strict: bool = True,
) -> xr.DataArray:
	"""
	Compute exceedance probability in [0, 1] for condition da > threshold.

	Set strict=False for >= threshold.
	"""
	mask = da > threshold if strict else da >= threshold
	return mask.mean(dim=member_dim, skipna=True)


def probability_in_range(
	da: xr.DataArray,
	lower: float,
	upper: float,
	member_dim: str = "member",
	include_lower: bool = True,
	include_upper: bool = False,
) -> xr.DataArray:
	"""Probability for values lying within [lower, upper) by default."""
	left = da >= lower if include_lower else da > lower
	right = da <= upper if include_upper else da < upper
	return (left & right).mean(dim=member_dim, skipna=True)


def exceedance_fraction(
	da: xr.DataArray,
	threshold: float,
	member_dim: str = "member",
	strict: bool = True,
) -> xr.DataArray:
	"""
	Fraction of members exceeding threshold, as percent [0, 100].

	Useful for decile-style displays (e.g. "60% of members >= 0.01 in").
	"""
	return 100.0 * probability_exceedance(
		da=da,
		threshold=threshold,
		member_dim=member_dim,
		strict=strict,
	)


def decile_membership(
	da: xr.DataArray,
	decile: int,
	member_dim: str = "member",
) -> xr.DataArray:
	"""
	Return the decile threshold field along the member dimension.

	Example:
		decile=6 returns the 60th percentile value at each grid point.
	"""
	if decile < 1 or decile > 10:
		raise ValueError("decile must be in [1, 10]")
	q = decile / 10.0
	return _ordered_quantile(da=da, q=q, dim=member_dim)


def fraction_exceedance_mask(
	da: xr.DataArray,
	threshold: float,
	required_fraction: float,
	member_dim: str = "member",
	strict: bool = False,
) -> xr.DataArray:
	"""
	Boolean mask where at least `required_fraction` of members exceed threshold.

	Example:
		required_fraction=0.6, threshold=0.01 gives where 60% of members
		have precip >= 0.01.
	"""
	if required_fraction < 0.0 or required_fraction > 1.0:
		raise ValueError("required_fraction must be in [0, 1]")
	frac = probability_exceedance(
		da=da,
		threshold=threshold,
		member_dim=member_dim,
		strict=strict,
	)
	return frac >= required_fraction


# -----------------------------------------------------------------------------
# Time-window and neighborhood probabilities
# -----------------------------------------------------------------------------


def rolling_window_max(
	da: xr.DataArray,
	window: int,
	time_dim: str = "time",
	min_periods: int | None = None,
) -> xr.DataArray:
	"""
	Rolling maximum along time (lazy for dask-backed arrays).

	Parameters
	----------
	window:
		Number of time steps in the rolling window.
	min_periods:
		Defaults to window (full-window requirement).
	"""
	if window < 1:
		raise ValueError("window must be >= 1")

	required = window if min_periods is None else min_periods
	return da.rolling({time_dim: window}, min_periods=required).max()


def neighborhood_max(
	da: xr.DataArray,
	radius_x: int,
	radius_y: int,
	x_dim: str = "x",
	y_dim: str = "y",
	min_periods: int = 1,
) -> xr.DataArray:
	"""
	Max filter over a rectangular neighborhood using rolling windows.

	This approximates neighborhood probabilities efficiently with xarray only.
	Window sizes are (2*radius + 1).
	"""
	if radius_x < 0 or radius_y < 0:
		raise ValueError("radius_x and radius_y must be >= 0")

	wx = (2 * radius_x) + 1
	wy = (2 * radius_y) + 1

	out = da.rolling({x_dim: wx}, center=True, min_periods=min_periods).max()
	out = out.rolling({y_dim: wy}, center=True, min_periods=min_periods).max()
	return out


def neighborhood_probability_exceedance(
	da: xr.DataArray,
	threshold: float,
	radius_x: int,
	radius_y: int,
	member_dim: str = "member",
	strict: bool = True,
	x_dim: str = "x",
	y_dim: str = "y",
) -> xr.DataArray:
	"""
	Neighborhood probability that any grid point in neighborhood exceeds threshold.

	Steps:
	1) binary threshold by member/time/grid
	2) neighborhood max on binary mask
	3) average over members -> probability
	"""
	binary = (da > threshold) if strict else (da >= threshold)
	binary = binary.astype("float32")
	nh = neighborhood_max(binary, radius_x=radius_x, radius_y=radius_y, x_dim=x_dim, y_dim=y_dim)
	return nh.mean(dim=member_dim, skipna=True)


def neighborhood_probability_time_window(
	da: xr.DataArray,
	threshold: float,
	time_window_steps: int,
	radius_x: int,
	radius_y: int,
	member_dim: str = "member",
	time_dim: str = "time",
	x_dim: str = "x",
	y_dim: str = "y",
	strict: bool = True,
) -> xr.DataArray:
	"""
	Neighborhood probability in a rolling time window.

	Example use: UH > 75 in 1-hour, 4-hour, and 24-hour windows.
	"""
	threshold_binary = (da > threshold) if strict else (da >= threshold)
	time_agg = rolling_window_max(
		threshold_binary.astype("float32"),
		window=time_window_steps,
		time_dim=time_dim,
		min_periods=time_window_steps,
	)
	nh = neighborhood_max(
		time_agg,
		radius_x=radius_x,
		radius_y=radius_y,
		x_dim=x_dim,
		y_dim=y_dim,
	)
	return nh.mean(dim=member_dim, skipna=True)


# -----------------------------------------------------------------------------
# PMM / LPMM
# -----------------------------------------------------------------------------


def probability_matched_mean(
	da: xr.DataArray,
	member_dim: str = "member",
	y_dim: str | None = None,
	x_dim: str | None = None,
) -> xr.DataArray:
	"""
	Probability Matched Mean (PMM).

	Algorithm (per time/level slice when those dims exist):
	- Compute ensemble mean field M
	- Flatten M and sort to obtain rank ordering of mean field
	- Flatten all member values and sort distribution D
	- Replace ranks of M with values from D at same quantile ranks

	Preserves spatial coherence of ensemble mean while restoring ensemble
	amplitude distribution.
	"""
	if member_dim not in da.dims:
		raise ValueError(f"member_dim '{member_dim}' not found in dims {da.dims}")

	resolved_y, resolved_x = _iter_xy_dims(da, y_dim=y_dim, x_dim=x_dim)
	mean_da = da.mean(dim=member_dim, skipna=True)

	def _pmm_core(member_field: np.ndarray) -> np.ndarray:
		# member_field shape: (n_member, ny, nx)
		n_member = member_field.shape[0]
		spatial_shape = member_field.shape[1:]
		n_points = int(np.prod(spatial_shape))

		mean_flat = np.nanmean(member_field, axis=0).reshape(-1)
		all_flat = member_field.reshape(n_member * n_points)

		valid_mean = np.isfinite(mean_flat)
		valid_all = np.isfinite(all_flat)

		if valid_mean.sum() == 0 or valid_all.sum() == 0:
			return np.full(spatial_shape, np.nan, dtype=np.float32)

		# Rank order of mean field (valid points only)
		mean_vals = mean_flat[valid_mean]
		mean_rank_idx = np.argsort(mean_vals)

		# Sorted pool of member values
		dist_vals = np.sort(all_flat[valid_all])

		# Quantile mapping from mean rank to distribution rank
		n_mean = mean_vals.size
		n_dist = dist_vals.size

		if n_mean == 1:
			mapped = np.array([dist_vals[n_dist // 2]], dtype=np.float64)
		else:
			q = np.arange(n_mean, dtype=np.float64) / (n_mean - 1)
			d_idx = np.clip(np.round(q * (n_dist - 1)).astype(np.int64), 0, n_dist - 1)
			mapped = dist_vals[d_idx]

		mean_matched = np.full_like(mean_vals, np.nan, dtype=np.float64)
		mean_matched[mean_rank_idx] = mapped

		out_flat = np.full_like(mean_flat, np.nan, dtype=np.float64)
		out_flat[valid_mean] = mean_matched
		return out_flat.reshape(spatial_shape).astype(np.float32)

	pmm = xr.apply_ufunc(
		_pmm_core,
		da,
		input_core_dims=[[member_dim, resolved_y, resolved_x]],
		output_core_dims=[[resolved_y, resolved_x]],
		vectorize=True,
		dask="parallelized",
		dask_gufunc_kwargs={"allow_rechunk": True},
		output_dtypes=[np.float32],
	)

	# Align coords/attrs with ensemble mean structure
	pmm = pmm.assign_coords({k: v for k, v in mean_da.coords.items() if k in pmm.dims})
	pmm.attrs.update(mean_da.attrs)
	pmm.attrs["postprocess"] = "PMM"
	return pmm


def localized_probability_matched_mean(
	da: xr.DataArray,
	member_dim: str = "member",
	x_dim: str = "x",
	y_dim: str = "y",
	radius_x: int = 10,
	radius_y: int = 10,
) -> xr.DataArray:
	"""
	Localized PMM (LPMM) approximation.

	Implementation strategy:
	- Start from PMM
	- Blend PMM toward local member maxima envelope to retain localized extremes
	- Blend weight determined by local ensemble spread normalized by robust scale

	This provides a practical, dask-friendly LPMM surrogate without expensive
	per-gridpoint neighborhood rank remapping.
	"""
	pmm = probability_matched_mean(
		da=da,
		member_dim=member_dim,
		y_dim=y_dim,
		x_dim=x_dim,
	)
	local_max = neighborhood_max(
		da.max(dim=member_dim, skipna=True),
		radius_x=radius_x,
		radius_y=radius_y,
		x_dim=x_dim,
		y_dim=y_dim,
	)

	spread = da.std(dim=member_dim, skipna=True)
	robust_scale = _ordered_quantile(spread, 0.95, dim=x_dim)
	robust_scale = _ordered_quantile(robust_scale, 0.95, dim=y_dim)

	# Guard against divide-by-zero with tiny epsilon
	weight = xr.where(robust_scale > 0, spread / (robust_scale + 1.0e-6), 0.0)
	weight = xr.where(weight > 1.0, 1.0, xr.where(weight < 0.0, 0.0, weight))

	lpmm = (1.0 - weight) * pmm + weight * local_max
	lpmm.attrs.update(pmm.attrs)
	lpmm.attrs["postprocess"] = "LPMM"
	lpmm.attrs["lpmm_radius_x"] = int(radius_x)
	lpmm.attrs["lpmm_radius_y"] = int(radius_y)
	return lpmm.astype(np.float32)


# -----------------------------------------------------------------------------
# Spaghetti contours
# -----------------------------------------------------------------------------


def spaghetti_contour_mask(
	da: xr.DataArray,
	contour_value: float,
	member_dim: str = "member",
	tolerance: float = 0.25,
) -> xr.DataArray:
	"""
	Fast contour-neighborhood mask for spaghetti-like products.

	Returns a per-member mask where values near contour level are 1.
	Use with contour extraction tooling for line geometry generation.
	"""
	if tolerance <= 0:
		raise ValueError("tolerance must be > 0")
	lower = contour_value - tolerance
	upper = contour_value + tolerance
	return ((da >= lower) & (da <= upper)).astype("uint8")


def spaghetti_probability_band(
	da: xr.DataArray,
	contour_value: float,
	member_dim: str = "member",
	tolerance: float = 0.25,
) -> xr.DataArray:
	"""
	Fraction of members with values in a narrow contour-centered band.

	Useful as a probabilistic spaghetti proxy.
	"""
	mask = spaghetti_contour_mask(
		da=da,
		contour_value=contour_value,
		member_dim=member_dim,
		tolerance=tolerance,
	)
	return mask.mean(dim=member_dim, skipna=True)


def extract_spaghetti_contours(
	da: xr.DataArray,
	contour_value: float,
	member_dim: str = "member",
	time_dim: str = "time",
	x_coord: str = "lon",
	y_coord: str = "lat",
) -> list[dict]:
	"""
	Extract contour lines by member/time using matplotlib, returning plain
	Python structures suitable for downstream encoding (e.g., AMBP).

	Output entries:
	{
	  "member": <member value>,
	  "time": <time value>,
	  "segments": [
		  [[x1, y1], [x2, y2], ...],
		  ...
	  ]
	}

	Notes:
	- This function computes selected slices to numpy arrays; it is not purely
	  lazy and should be used for rendering-ready products.
	"""
	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise ImportError(
			"matplotlib is required for contour extraction. Install with: pip install matplotlib"
		) from exc

	if member_dim not in da.dims:
		raise ValueError(f"member_dim '{member_dim}' not found in dims {da.dims}")
	if time_dim not in da.dims:
		raise ValueError(f"time_dim '{time_dim}' not found in dims {da.dims}")

	if x_coord not in da.coords or y_coord not in da.coords:
		raise ValueError(f"Expected coords '{x_coord}' and '{y_coord}' in DataArray coords")

	xvals = da.coords[x_coord].values
	yvals = da.coords[y_coord].values

	results: list[dict] = []
	for t in da.coords[time_dim].values:
		for m in da.coords[member_dim].values:
			fld = da.sel({time_dim: t, member_dim: m}).values
			cs = plt.contour(xvals, yvals, fld, levels=[contour_value])

			segments: list[list[list[float]]] = []
			for col in cs.collections:
				for path in col.get_paths():
					vertices = path.vertices
					if vertices.shape[0] < 2:
						continue
					segments.append(vertices.astype(np.float32).tolist())

			plt.close()
			results.append(
				{
					"member": _coerce_scalar(m),
					"time": _coerce_scalar(t),
					"segments": segments,
				}
			)

	return results


def _coerce_scalar(value):
	if isinstance(value, np.generic):
		return value.item()
	return value


# -----------------------------------------------------------------------------
# Dataset builders and orchestrators
# -----------------------------------------------------------------------------


def add_grid_metadata_attrs(
	ds: xr.Dataset,
	*,
	y_coord_candidates: Sequence[str] = DEFAULT_Y_CANDIDATES,
	x_coord_candidates: Sequence[str] = DEFAULT_X_CANDIDATES,
	normalize_lon: bool = False,
) -> xr.Dataset:
	"""
	Add common grid metadata attrs to dataset-level attributes.

	This is useful before writing Zarr so consumers can quickly discover
	grid extents/resolution.
	"""
	out = ds.copy()

	y_name = _find_coord_name(out, y_coord_candidates)
	x_name = _find_coord_name(out, x_coord_candidates)

	if y_name is None or x_name is None:
		return out

	yv = np.asarray(out[y_name].values)
	xv = np.asarray(out[x_name].values)

	if normalize_lon:
		xv = _normalize_longitudes(xv)

	if yv.ndim == 1 and xv.ndim == 1:
		dy = abs(float(yv[1] - yv[0])) if yv.size > 1 else np.nan
		dx = abs(float(xv[1] - xv[0])) if xv.size > 1 else np.nan
	else:
		# fallback for 2D lat/lon; estimate corner deltas if possible
		dy = abs(float(yv[1, 0] - yv[0, 0])) if yv.shape[0] > 1 else np.nan
		dx = abs(float(xv[0, 1] - xv[0, 0])) if xv.shape[1] > 1 else np.nan

	out.attrs.update(
		{
			"grid_y_coord": str(y_name),
			"grid_x_coord": str(x_name),
			"grid_lat_min": float(np.nanmin(yv)),
			"grid_lat_max": float(np.nanmax(yv)),
			"grid_lon_min": float(np.nanmin(xv)),
			"grid_lon_max": float(np.nanmax(xv)),
			"grid_dy": float(dy) if np.isfinite(dy) else "unknown",
			"grid_dx": float(dx) if np.isfinite(dx) else "unknown",
		}
	)
	return out


def apply_ensemble_diagnostics(
	ds: xr.Dataset,
	variable: str,
	*,
	member_dim: str | None = None,
	time_dim: str | None = None,
	y_dim: str | None = None,
	x_dim: str | None = None,
	thresholds: Mapping[str, float] | None = None,
	percentile_probs: Sequence[float] | None = None,
	neighborhood_probability_requests: Sequence[dict] | None = None,
	contour_band_requests: Sequence[dict] | None = None,
	include_pmm: bool = True,
	include_lpmm: bool = True,
	lpmm_radius_x: int = 10,
	lpmm_radius_y: int = 10,
) -> xr.Dataset:
	"""
	Build a diagnostics Dataset for one ensemble variable.

	Parameters
	----------
	ds:
		Input Dataset containing `variable` with ensemble dimension.
	variable:
		Target variable name in `ds`.
	thresholds:
		Dict of name -> threshold for exceedance probabilities.
		Example: {"stp_gt_1": 1.0}
	percentile_probs:
		Optional quantiles in [0,1] to emit as fields.
	neighborhood_probability_requests:
		Each dict may contain:
			{
			  "name": "uh75_4h_nprob",
			  "threshold": 75.0,
			  "time_window_steps": 4,
			  "radius_x": 8,
			  "radius_y": 8,
			  "strict": True
			}
	contour_band_requests:
		Each dict may contain:
			{
			  "name": "dpt70_spag_prob",
			  "contour_value": 70.0,
			  "tolerance": 0.5
			}
	"""
	if variable not in ds:
		raise KeyError(f"Variable '{variable}' not found in dataset")

	da = ds[variable]
	dims = infer_ensemble_dims(
		da,
		member_dim=member_dim,
		time_dim=time_dim,
		y_dim=y_dim,
		x_dim=x_dim,
	)

	out = xr.Dataset()

	# Baseline fields
	out[f"{variable}_mean"] = mean_field(da, member_dim=dims.member_dim)
	out[f"{variable}_spread"] = spread_field(da, member_dim=dims.member_dim)
	out[f"{variable}_max"] = max_field(da, member_dim=dims.member_dim)
	out[f"{variable}_min"] = min_field(da, member_dim=dims.member_dim)

	# Threshold probabilities
	if thresholds:
		for label, threshold in thresholds.items():
			out[f"prob_{label}"] = probability_exceedance(
				da,
				threshold=float(threshold),
				member_dim=dims.member_dim,
				strict=True,
			)

	# Percentiles / decile-like outputs
	if percentile_probs:
		for q in percentile_probs:
			if not (0.0 <= q <= 1.0):
				raise ValueError(f"percentile q must be in [0,1], got {q}")
			pct_name = int(round(q * 100.0))
			out[f"{variable}_p{pct_name:02d}"] = _ordered_quantile(
				da,
				q=float(q),
				dim=dims.member_dim,
			)

	# Neighborhood probs
	if neighborhood_probability_requests:
		if dims.time_dim is None:
			raise ValueError(
				"No time dimension could be inferred; set time_dim explicitly or "
				"omit neighborhood_probability_requests."
			)
		for req in neighborhood_probability_requests:
			name = str(req["name"])
			thr = float(req["threshold"])
			tw = int(req["time_window_steps"])
			rx = int(req["radius_x"])
			ry = int(req["radius_y"])
			strict = bool(req.get("strict", True))

			out[name] = neighborhood_probability_time_window(
				da=da,
				threshold=thr,
				time_window_steps=tw,
				radius_x=rx,
				radius_y=ry,
				member_dim=dims.member_dim,
				time_dim=dims.time_dim,
				x_dim=dims.x_dim,
				y_dim=dims.y_dim,
				strict=strict,
			)

	# Spaghetti contour probability bands
	if contour_band_requests:
		for req in contour_band_requests:
			name = str(req["name"])
			contour_value = float(req["contour_value"])
			tol = float(req.get("tolerance", 0.25))
			out[name] = spaghetti_probability_band(
				da=da,
				contour_value=contour_value,
				member_dim=dims.member_dim,
				tolerance=tol,
			)

	# PMM + LPMM
	if include_pmm:
		out[f"{variable}_pmm"] = probability_matched_mean(
			da=da,
			member_dim=dims.member_dim,
			y_dim=dims.y_dim,
			x_dim=dims.x_dim,
		)
	if include_lpmm:
		out[f"{variable}_lpmm"] = localized_probability_matched_mean(
			da=da,
			member_dim=dims.member_dim,
			x_dim=dims.x_dim,
			y_dim=dims.y_dim,
			radius_x=lpmm_radius_x,
			radius_y=lpmm_radius_y,
		)

	out.attrs.update(
		{
			"source_variable": variable,
			"member_dim": dims.member_dim,
			"time_dim": dims.time_dim,
			"x_dim": dims.x_dim,
			"y_dim": dims.y_dim,
		}
	)
	return out


def write_dataset_to_zarr(
	ds: xr.Dataset,
	zarr_path: str | Path,
	*,
	mode: str = "w",
	consolidated: bool = True,
	compute: bool = True,
) -> None:
	"""
	Write Dataset to Zarr.

	Keeps lazy graph intact until compute=True triggers execution.
	"""
	ds.to_zarr(
		str(zarr_path),
		mode=mode,
		consolidated=consolidated,
		compute=compute,
	)


def convert_dataset_to_zarr(
	ds: xr.Dataset,
	zarr_path: str | Path,
	*,
	include_grid_metadata: bool = True,
	normalize_lon: bool = False,
	mode: str = "w",
	consolidated: bool = True,
	compute: bool = True,
) -> xr.Dataset:
	"""
	Convert/load-agnostic Dataset into Zarr with optional grid metadata attrs.
	"""
	out = ds
	if include_grid_metadata:
		out = add_grid_metadata_attrs(out, normalize_lon=normalize_lon)
	write_dataset_to_zarr(
		out,
		zarr_path=zarr_path,
		mode=mode,
		consolidated=consolidated,
		compute=compute,
	)
	return out


def process_ensemble_to_zarr(
	ds: xr.Dataset,
	variable: str,
	output_zarr_path: str | Path,
	*,
	thresholds: Mapping[str, float] | None = None,
	percentile_probs: Sequence[float] | None = None,
	neighborhood_probability_requests: Sequence[dict] | None = None,
	contour_band_requests: Sequence[dict] | None = None,
	include_pmm: bool = True,
	include_lpmm: bool = True,
	lpmm_radius_x: int = 10,
	lpmm_radius_y: int = 10,
	include_grid_metadata: bool = True,
	normalize_lon: bool = False,
	mode: str = "w",
	consolidated: bool = True,
	compute: bool = True,
) -> xr.Dataset:
	"""
	End-to-end convenience helper:
	  - compute ensemble diagnostics for a variable
	  - merge into a dataset
	  - write to Zarr

	Returns the diagnostics dataset (lazy unless compute=True and eager ops invoked).
	"""
	diag = apply_ensemble_diagnostics(
		ds=ds,
		variable=variable,
		thresholds=thresholds,
		percentile_probs=percentile_probs,
		neighborhood_probability_requests=neighborhood_probability_requests,
		contour_band_requests=contour_band_requests,
		include_pmm=include_pmm,
		include_lpmm=include_lpmm,
		lpmm_radius_x=lpmm_radius_x,
		lpmm_radius_y=lpmm_radius_y,
	)

	if include_grid_metadata:
		diag = add_grid_metadata_attrs(diag, normalize_lon=normalize_lon)

	write_dataset_to_zarr(
		diag,
		zarr_path=output_zarr_path,
		mode=mode,
		consolidated=consolidated,
		compute=compute,
	)
	return diag


# -----------------------------------------------------------------------------
# Example usage entry point
# -----------------------------------------------------------------------------


def example_usage() -> str:
	"""
	Return a short usage snippet for interactive/dev reference.
	"""
	return (
		"import xarray as xr\n"
		"from backend.processing.ens_postprocessing import process_ensemble_to_zarr\n\n"
		"ds = xr.open_dataset('input.nc', chunks={'time': 1, 'member': 5})\n"
		"diag = process_ensemble_to_zarr(\n"
		"    ds=ds,\n"
		"    variable='uh',\n"
		"    output_zarr_path='uh_diag.zarr',\n"
		"    thresholds={'uh_gt_75': 75.0, 'stp_gt_1': 1.0},\n"
		"    percentile_probs=[0.1, 0.5, 0.9],\n"
		"    neighborhood_probability_requests=[\n"
		"        {'name': 'uh75_1h_nprob', 'threshold': 75.0, 'time_window_steps': 1, 'radius_x': 8, 'radius_y': 8},\n"
		"        {'name': 'uh75_4h_nprob', 'threshold': 75.0, 'time_window_steps': 4, 'radius_x': 8, 'radius_y': 8},\n"
		"        {'name': 'uh75_24h_nprob', 'threshold': 75.0, 'time_window_steps': 24, 'radius_x': 8, 'radius_y': 8},\n"
		"    ],\n"
		"    contour_band_requests=[\n"
		"        {'name': 'dpt70_spag_prob', 'contour_value': 70.0, 'tolerance': 0.5},\n"
		"    ],\n"
		")\n"
	)


__all__ = [
	"EnsembleDims",
	"infer_ensemble_dims",
	"mean_field",
	"spread_field",
	"max_field",
	"min_field",
	"probability_exceedance",
	"probability_in_range",
	"exceedance_fraction",
	"decile_membership",
	"fraction_exceedance_mask",
	"rolling_window_max",
	"neighborhood_max",
	"neighborhood_probability_exceedance",
	"neighborhood_probability_time_window",
	"probability_matched_mean",
	"localized_probability_matched_mean",
	"spaghetti_contour_mask",
	"spaghetti_probability_band",
	"extract_spaghetti_contours",
	"add_grid_metadata_attrs",
	"apply_ensemble_diagnostics",
	"write_dataset_to_zarr",
	"convert_dataset_to_zarr",
	"process_ensemble_to_zarr",
	"example_usage",
]
