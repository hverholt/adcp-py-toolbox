from tempfile import TemporaryDirectory
from pathlib import Path
from mhkit import dolfyn
import xarray as xr
import numpy as np


def read(ifile, drop_ids=(23,), **kwargs):
    """
    Read a Nortek AD2CP file with unsupported record types removed.

    The filtered AD2CP file is created only temporarily and is
    automatically deleted after reading.

    Parameters
    ----------
    ifile : str or pathlib.Path
        Path to the original AD2CP file.

    drop_ids : iterable of int, optional
        AD2CP record IDs to exclude before reading.
        Default is (23,), corresponding to bottom-track V9 records.

    **kwargs
        Additional keyword arguments passed to dolfyn.read().

    Returns
    -------
    ds : xarray.Dataset
        Dataset returned by dolfyn.read().
    """

    ifile = Path(ifile)
    drop_ids = set(drop_ids)

    # Open a temporary directory
    with TemporaryDirectory() as tmpdir:

        tmpfile = Path(tmpdir) / ifile.name

        # Create temporary filtered AD2CP file
        with open(ifile, "rb") as fin, open(tmpfile, "wb") as fout:

            while True:
                header_start = fin.tell()
                h = fin.read(4)

                if not h:
                    break

                if len(h) < 4:
                    raise IOError(
                        f"Incomplete header at byte {header_start}"
                    )

                sync, hsz, record_id, family = h

                if sync != 0xA5 or hsz not in (10, 12):
                    raise IOError(
                        f"Invalid AD2CP header at byte {header_start}"
                    )

                rest = fin.read(hsz - 4)

                if len(rest) != hsz - 4:
                    raise IOError(
                        f"Incomplete header at byte {header_start}"
                    )

                if hsz == 10:
                    size = int.from_bytes(
                        rest[0:2],
                        "little",
                    )
                else:
                    size = int.from_bytes(
                        rest[0:4],
                        "little",
                    )

                payload = fin.read(size)

                if len(payload) != size:
                    raise IOError(
                        f"Incomplete record at byte {header_start}"
                    )

                # Write only supported records
                if record_id not in drop_ids:
                    fout.write(h)
                    fout.write(rest)
                    fout.write(payload)

        # Read temporary file with DOLfYN
        ds = dolfyn.read(
            str(tmpfile),
            rebuild_index=True,
            **kwargs,
        )

    return ds


def standardize(ds, coord_sys=None):
    """
    Format an MHKiT/DOLfYN AD2CP dataset to telemetry-style naming.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset returned by dolfyn.read().
    coord_sys : {'BEAM', 'XYZ', 'ENU', None}
        Desired velocity coordinate system.
        None keeps the coordinate system already present in ds.

    Returns
    -------
    out : xarray.Dataset
        Dataset using telemetry-style variable names.
    """

    # Coordinate system
    current = ds.attrs["coord_sys"].lower()

    coord_map = {
        "BEAM": "beam",
        "XYZ": "inst",
        "ENU": "earth",
    }
    reverse_coord_map = {
        "beam": "BEAM",
        "inst": "XYZ",
        "earth": "ENU",
    }

    if coord_sys is None:
        target = current
    else:
        coord_sys = coord_sys.upper()

        if coord_sys not in coord_map:
            raise ValueError(
                "coord_sys must be one of "
                "{'BEAM', 'XYZ', 'ENU', None}"
            )

        target = coord_map[coord_sys]

    # Rotate velocities if required
    if target != current:
        ds = dolfyn.rotate2(
            ds,
            target,
            inplace=False,
        )

    CY = reverse_coord_map[target]

    # Create output dataset
    n_cells = ds.sizes["range"]
    n_time = ds.sizes["time"]

    out = xr.Dataset(
        coords={
            "TIME": ds["time"].values,
            "CN": np.arange(1, n_cells + 1),
        }
    )

    # Cell position in meters (range)
    out["CP"] = (
        ("TIME", "CN"),
        np.broadcast_to(
            ds["range"].values,
            (n_time, n_cells),
        ).copy()
    )

    # Sensors
    out["H"]  = ("TIME", ds["heading"].values)
    out["PI"] = ("TIME", ds["pitch"].values)
    out["R"]  = ("TIME", ds["roll"].values)
    out["P"]  = ("TIME", ds["pressure"].values)
    out["T"]  = ("TIME", ds["temp"].values)

    out["CY"] = xr.DataArray(CY)

    # Velocity
    vel = ds["vel"].transpose("dir", "time", "range").values

    if CY == "BEAM":
        out["V1"] = (("TIME", "CN"), vel[0])
        out["V2"] = (("TIME", "CN"), vel[1])
        out["V3"] = (("TIME", "CN"), vel[2])
        out["V4"] = (("TIME", "CN"), vel[3])

    elif CY == "XYZ":
        out["VX"]  = (("TIME", "CN"), vel[0])
        out["VY"]  = (("TIME", "CN"), vel[1])
        out["VZ"]  = (("TIME", "CN"), vel[2])
        out["VZ2"] = (("TIME", "CN"), vel[3])

    elif CY == "ENU":
        out["VE"]  = (("TIME", "CN"), vel[0])
        out["VN"]  = (("TIME", "CN"), vel[1])
        out["VU"]  = (("TIME", "CN"), vel[2])
        out["VU2"] = (("TIME", "CN"), vel[3])

    # Correlation
    corr = ds["corr"].transpose("beam", "time", "range").values
    
    out["C1"] = (("TIME", "CN"), corr[0])
    out["C2"] = (("TIME", "CN"), corr[1])
    out["C3"] = (("TIME", "CN"), corr[2])
    out["C4"] = (("TIME", "CN"), corr[3])
    
    # Amplitude
    amp = ds["amp"].transpose("beam", "time", "range").values
    
    out["A1"] = (("TIME", "CN"), amp[0])
    out["A2"] = (("TIME", "CN"), amp[1])
    out["A3"] = (("TIME", "CN"), amp[2])
    out["A4"] = (("TIME", "CN"), amp[3])
    
    return out
