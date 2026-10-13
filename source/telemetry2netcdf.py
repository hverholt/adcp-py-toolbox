import argparse
import re
import os
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


def read_tm_file_to_dict(ifile):
    with open(ifile, 'r') as f:
        lines = f.readlines()
    data_dict = defaultdict(lambda: defaultdict(list))
    
    # Interpret lines
    for line in lines:
        match = re.match(r'\$(\w+),(.*)\*(\w+)', line.strip())
        if not match:
            continue
        identifier, fields, checksum = match.group(1), match.group(2), match.group(3)

        # Only process actual telemetry records
        if identifier not in ("PNORI", "PNORS", "PNORC"):
            continue
        
        # If telemetry contains keys, use keys
        if '=' in fields:
            for field in fields.split(','):
                key, value = field.split('=', 1)
                data_dict[identifier][key].append(value)
        
        # If telemetry contains no keys, use the index as keys
        else:
            for i, field in enumerate(fields.split(',')):
                data_dict[identifier][i].append(field)

    return data_dict


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
            df = df.reset_index(drop=True).set_index(indexes)
            ds_part = df.to_xarray()
            ds = ds.merge(ds_part)

        # Add each column to the xr.Dataset
        else:
            for col in df.columns:
                if len(df[col]) == len(ds['TIME']):
                    ds[col] = (('TIME'), df[col])
                else:
                    ds[col] = df[col]

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
    #filename = os.path.basename(ifile).rpartition('.')[0]
    #ofile = f'data/formatted/{filename}.nc'
    ds.to_netcdf(ofile)


def telemetry_to_netcdf(ifile, ofile, data_format, error_values=error_values, filter_errors=False):
    tm_data_dict = read_tm_file_to_dict(ifile)
    
    if data_format == 100:
        data_format_dict = NMEA_FORMAT_100
    elif data_format == 101 or data_format == 102:
        data_format_dict = NMEA_FORMAT_101_102
    else:
        raise ValueError(f'{repr(data_format)} is not a valid value for data_format; available values are: 100, 101, 102')

    tm_data_dict = assign_keys(
        tm_data_dict,
        data_format_dict
    )
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


if __name__ == '__main__':
    data_format = 102
    parser = argparse.ArgumentParser(description='Convert telemetry TXT file to NetCDF.')
    #parser.add_argument('data_format', help='Telemetry NMEA data format')
    parser.add_argument('input_txt', help='Path to input TXT file')
    parser.add_argument('output_nc', help='Path to output NetCDF file')
    args = parser.parse_args()
    
    ofile = telemetry_to_netcdf(args.input_txt, args.output_nc, data_format, filter_errors=True)
    
    print(f'Conversion completed. File saved to {ofile}')
