import numpy as np

def rotate_xyz2neu(xyz, heading=0, pitch=0, roll=0, declination=0):
    """
    Converts XYZ data into NEU (North, East, Up) data using given rotational coordinates as defined
    by the Signature Series Operations Manual. 
        Declination must be given as an Eastward declination: -W/+E.
    """
    # Correct for declination
    heading += declination
    
    # Convert to radians
    h = np.deg2rad(heading)
    p = np.deg2rad(pitch)
    r = np.deg2rad(roll)
    
    Rz = np.array([
        [np.cos(h), -np.sin(h), 0],
        [np.sin(h),  np.cos(h), 0],
        [0,          0,         1]
    ]) 
    Ry = np.array([
        [ np.cos(p), 0,  np.sin(p)],
        [0,         -1,          0],
        [-np.sin(p), 0,  np.cos(p)],
    ])
    Rx = np.array([
        [1,         0,          0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r),  np.cos(r)],
    ])

    R = Rz @ Ry @ Rx

    return R @ np.asarray(xyz)


def rotate_neu2apu(neu, bearing=0):
    """
    Convert NEU (North, East, Up) data into APU (Along, Perpendicular, Up) data, relative to a given
    bearing, such as the bearing of a fjord for example. NEU must be given as a vstacked numpy array.
    """
    # Convert to radians
    b = np.deg2rad(bearing)

    R = np.array([
        [np.cos(b),  np.sin(b), 0],
        [np.sin(b), -np.cos(b), 0],
        [0,          0,         1]
    ])

    return R @ np.asarray(neu)

