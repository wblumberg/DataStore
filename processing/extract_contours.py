import xarray as xr
import numpy as np
import zarr
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

# Could switch to skimage, which will be faster (need map of pixel -> lat, lon)
# Add contour simplification (LineString from shapely can be simplified)
# Create a true distributed Dask
# Pre-project to web mercator?

def extract_contours_numpy(field, lons, lats, level_value):
    cs = plt.contour(lons, lats, field, levels=[level_value])

    contours = []
    for col in cs.collections:
        for path in col.get_paths():
            contours.append(path.vertices)

    plt.close()
    return contours

def contours_to_chunk(contours, member_idx, time_idx, seg_start):
    xs, ys = [], []
    mems, times, segs = [], [], []

    seg_id = seg_start

    for c in contours:
        if len(c) < 2:
            continue

        x = c[:, 0].astype("float32")
        y = c[:, 1].astype("float32")

        n = len(x)

        xs.append(x)
        ys.append(y)
        mems.append(np.full(n, member_idx, dtype="int16"))
        times.append(np.full(n, time_idx, dtype="int32"))
        segs.append(np.full(n, seg_id, dtype="int32"))

        seg_id += 1

    if len(xs) == 0:
        return None, seg_id

    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(mems),
        np.concatenate(times),
        np.concatenate(segs),
        seg_id
    )

@delayed
def process_member_time(field, lons, lats, level_value, m_idx, t_idx, seg_start):
    contours = extract_contours_numpy(field, lons, lats, level_value)

    result, seg_end = contours_to_chunk(
        contours, m_idx, t_idx, seg_start
    )

    return result, seg_end

def generate_contours_to_zarr(
    zarr_path,
    var_name="gh",
    pressure_level=500,
    contour_value=5400.0,
):
    # Open dataset lazily
    ds = xr.open_zarr(zarr_path, chunks={})

    da = ds[var_name].sel(level=pressure_level)

    lons = ds["lon"].values
    lats = ds["lat"].values

    tasks = []
    seg_counter = 0

    # Build Dask graph
    for t_idx, t in enumerate(da.time.values):
        for m_idx, m in enumerate(da.member.values):

            field = da.sel(time=t, member=m).values  # small slice

            task = process_member_time(
                field,
                lons,
                lats,
                contour_value,
                m_idx,
                t_idx,
                seg_counter
            )

            tasks.append(task)

            # NOTE: we increment a *rough* segment counter to avoid collisions
            seg_counter += 10000  # safe buffer

    # Execute in parallel
    with ProgressBar():
        results = compute(*tasks)

    # Flatten results
    X, Y, MEM, TIME, SEG = [], [], [], [], []

    for result, _ in results:
        if result is None:
            continue

        x, y, mem, tim, seg = result

        X.append(x)
        Y.append(y)
        MEM.append(mem)
        TIME.append(tim)
        SEG.append(seg)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    MEM = np.concatenate(MEM)
    TIME = np.concatenate(TIME)
    SEG = np.concatenate(SEG)

    # Write into EXISTING Zarr
    root = zarr.open(zarr_path, mode="a")

    group_path = f"contours/{var_name}/{pressure_level}/{int(contour_value)}"
    grp = root.require_group(group_path)

    grp.create_dataset("x", data=X, overwrite=True, chunks=(100000,))
    grp.create_dataset("y", data=Y, overwrite=True, chunks=(100000,))
    grp.create_dataset("member", data=MEM, overwrite=True, chunks=(100000,))
    grp.create_dataset("time", data=TIME, overwrite=True, chunks=(100000,))
    grp.create_dataset("segment_id", data=SEG, overwrite=True, chunks=(100000,))

    print(f"Contours written to: {group_path}")
