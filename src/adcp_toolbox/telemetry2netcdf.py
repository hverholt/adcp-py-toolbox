import argparse
import logging
import math
from datetime import datetime
from functools import lru_cache
import re
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from collections import defaultdict

# NMEA Format keys
NMEA_FORMAT_100 = dict(
    PNORI = ['IT','SN','NB','NC','BD','CS','CY'],
    PNORS = ['DATE','TIME','EC','SC','BV','SS','H','PI','R','P','T','AI1','AI2'],
    PNORC = ['DATE','TIME','CN','V1','V2','V3','V4','SPEED','DIR','A_UNIT','A1','A2','A3','A4','C1','C2','C3','C4']
)
NMEA_FORMAT_101_102 = dict(
    PNORI = ['IT','SN','NB','NC','BD','CS','CY'],
    PNORS = ['DATE','TIME','EC','SC','BV','SS','HSD','H','PI','PISD','R','RSD','P','PSD','T'],
    PNORC = ['DATE','TIME','CN','CP','V1','V2','V3','V4','A1','A2','A3','A4','C1','C2','C3','C4']
)

# Dtypes to be assigned to processed data
processed_data_dtypes = dict(
    TIME = np.datetime64,
    SN = str,
    CY = str,
    EC = str,
    SC = str,
    A_UNIT = str,
)

# Dtype fill value for unspecified dtypes in processed data
processed_data_dtype_fill = float

# Values to replace with nan in processed data
error_values = [-32.768,-9,-99,-999,-9999]


_sentence = re.compile(r"\$(PNORI|PNORS|PNORC)\d*,([^*$]+)\*([0-9a-fA-F]{2})")
_log = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _check_time(date, time):
    if len(date) != 6 or len(time) != 6:
        raise ValueError("invalid date/time width")
    datetime.strptime(date + time, "%m%d%y%H%M%S")


def _parse_row(kind, fields, layout, keyed):
    names = layout[kind]
    if len(fields) != len(names):
        raise ValueError(f"{len(fields)} fields, expected {len(names)}")
    row = {}
    for index, field in enumerate(fields):
        if keyed:
            if field.count("=") != 1:
                raise ValueError("invalid key=value field")
            key, value = field.split("=")
        else:
            key, value = names[index], field
        if key in row:
            raise ValueError(f"duplicate {key}")
        row[key] = value

    # Nortek names velocity components according to the coordinate system.
    expected = set(names)
    if kind == "PNORC":
        expected.difference_update(("V1", "V2", "V3", "V4"))
        components = set(row) - expected
        if len(components) != 4 or not all(re.fullmatch(r"V[A-Z0-9]+", k) for k in components):
            raise ValueError("invalid velocity component keys")
    if not expected.issubset(row):
        raise ValueError("missing required field names")
    if kind != "PNORI":
        _check_time(row["DATE"], row["TIME"])
    if kind == "PNORC" and (not row["CN"].isdigit() or int(row["CN"]) < 1):
        raise ValueError("invalid cell number")
    for key, value in row.items():
        if key in ("DATE", "TIME") or key in processed_data_dtypes:
            continue
        if value == "" or not math.isfinite(float(value)):
            row[key] = np.nan
    return row


def read_tm_file_to_dict(ifile, data_format=None):
    """Validate each whole record before committing its fields to columns."""
    columns = defaultdict(lambda: defaultdict(list))
    skipped = 0
    examples = []
    with open(ifile, encoding="ascii", errors="replace") as source:
        for number, line in enumerate(source, 1):
            line = line.strip()
            if not line.startswith(("$PNORI", "$PNORS", "$PNORC")):
                continue
            try:
                match = _sentence.fullmatch(line)
                if not match:
                    raise ValueError("incomplete or malformed sentence")
                kind, payload, _ = match.groups()
                fields = payload.split(",")
                keyed = "=" in payload
                layout = data_format
                if layout is None:
                    layout = (NMEA_FORMAT_100 if len(fields) == len(NMEA_FORMAT_100[kind])
                              else NMEA_FORMAT_101_102)
                row = _parse_row(kind, fields, layout, keyed)
                if not keyed and data_format is None:
                    row = dict(enumerate(row.values()))
                table = columns[kind]
                if table and table.keys() != row.keys():
                    raise ValueError("field names changed within file")
            except ValueError as exc:
                skipped += 1
                if len(examples) < 10:
                    examples.append(f"line {number}: {exc}")
                continue
            for key, value in row.items():
                table[key].append(value)
    if skipped:
        _log.warning("Skipped %d malformed telemetry records (%s)", skipped, "; ".join(examples))
    return columns


def assign_keys(data_dict, data_format):
    new_data_dict = {}

    for outer_key, inner_dict in data_dict.items():
        # Check if all inner keys are integers
        if all(isinstance(k, int) for k in inner_dict):
            mapped_dict = {data_format[outer_key][i]: v for i, v in inner_dict.items()}
        # Else leave as is
        else:
            mapped_dict = inner_dict

        new_data_dict[outer_key] = mapped_dict

    return new_data_dict


def dict_to_xrdataset(data_dict):
    def parse_time(df):
        df['TIME'] = pd.to_datetime(df['DATE']+' '+df['TIME'], format='%m%d%y %H%M%S')
        return df.drop(columns='DATE')

    # Create xr.Dataset
    ds = xr.Dataset()

    priority = ['C', 'S', 'I']
    ordered_keys = sorted(
        data_dict.keys(),
        key=lambda k: next((i for i, c in enumerate(priority) if c in k), len(priority))
    )
    for key in ordered_keys:
        data = data_dict[key]
        df = pd.DataFrame(data)

        if 'TIME' in df.columns and 'DATE' in df.columns:
            df = parse_time(df)

        indexes = []
        if 'TIME' in df.columns:
            indexes.append('TIME')
        if 'CN' in df.columns:
            indexes.append('CN')

        # Merge columns into xr.Dataset
        if indexes:
            if "CN" in indexes:
                df["CN"] = pd.to_numeric(df["CN"])
            repeated = df.duplicated(indexes, keep="first")
            if repeated.any():
                _log.warning("Ignoring %d repeated %s records; keeping first", repeated.sum(), key)
                df = df.loc[~repeated]
            df = df.reset_index(drop=True).set_index(indexes)
            ds_part = df.to_xarray()
            ds = ds.merge(ds_part, join="outer")

        # Add each column to the xr.Dataset
        else:
            for col in df.columns:
                if len(df[col]) == len(ds['TIME']):
                    ds[col] = (('TIME'), df[col])
                else:
                    ds[col] = ("PNORI_record", df[col].to_numpy())

    return ds


def assign_dtypes(ds, dtypes, fill=None):
    for col in ds.variables:
        dtype = dtypes.get(col, fill)

        if dtype is not None:
            try:
                ds[col] = ds[col].astype(dtype)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert variable '{col}' to {dtype}: {e}"
                ) from e


def remove_errors(ds, error_values):
    return ds.where(lambda v: ~v.isin(np.atleast_1d(error_values)))


def order_xrdataset_by_coords(ds):
    return ds.sortby(list(ds.coords))


def export_xrdataset_to_netcdf(ds, ofile):
    ds.to_netcdf(ofile)


def telemetry_to_netcdf(ifile, ofile, data_format, error_values=error_values, filter_errors=False):
    
    if data_format == 100:
        data_format_dict = NMEA_FORMAT_100
    elif data_format == 101 or data_format == 102:
        data_format_dict = NMEA_FORMAT_101_102
    else:
        raise ValueError(f'{repr(data_format)} is not a valid value for data_format; available values are: 100, 101, 102')

    tm_data_dict = read_tm_file_to_dict(ifile, data_format_dict)
    if not tm_data_dict.get("PNORC") and not tm_data_dict.get("PNORS"):
        raise ValueError("No usable timestamped telemetry records.")
    tm_ds = dict_to_xrdataset(tm_data_dict)

    assign_dtypes(
        tm_ds,
        dtypes=processed_data_dtypes,
        fill=processed_data_dtype_fill
    )

    if filter_errors:
        tm_ds = remove_errors(tm_ds, error_values)

    tm_ds = order_xrdataset_by_coords(tm_ds)

    export_xrdataset_to_netcdf(
        tm_ds,
        ofile
    )
    return ofile


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for Nortek Signature telemetry conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Nortek Signature ADCP telemetry TXT files to NetCDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_txt", type=Path,
        help="Input telemetry text file.",
    )
    parser.add_argument(
        "output_nc", nargs="?", type=Path,
        help="Output NetCDF file (defaults to the input filename with a .nc suffix).",
    )
    parser.add_argument(
        "--data-format", type=int, choices=(100, 101, 102), default=102,
        help="Nortek telemetry data format.",
    )
    parser.add_argument(
        "--filter-errors", action=argparse.BooleanOptionalAction, default=True,
        help="Replace known invalid measurement values with NaN.",
    )
    args = parser.parse_args(argv)

    input_path = args.input_txt.expanduser()
    output_path = (args.output_nc or input_path.with_suffix(".nc")).expanduser()

    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        parser.error("Input and output paths must differ.")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_to_netcdf(
            ifile=input_path,
            ofile=output_path,
            data_format=args.data_format,
            filter_errors=args.filter_errors,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")

    print(f"Saved NetCDF: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
