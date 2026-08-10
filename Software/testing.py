import numpy as np
import math


def generate_fm_array_from_xyz(xyz_points):
    """
    Generate an fm_array from manually supplied tissue-center XYZ positions.

    Spatial convention:
        - X increases from left to right
        - Y decreases from top to bottom

    Therefore:
        upper-left  = lowest X, highest Y
        lower-right = highest X, lowest Y

    Input tissue centers are sorted by:
        1. Highest Y first
        2. Lowest X first

    Each tissue center generates a 3x3 tile montage with 10% overlap.

    Parameters
    ----------
    xyz_points : list of tuple
        List of (X, Y, DAPI_Z) tissue-center coordinates.

        Example:
            [
                (10000.0, 5000.0, 1245.3),
                (25000.0, 18000.0, 1251.8),
                (41000.0, 7000.0, 1247.1)
            ]

    Returns
    -------
    fm_array : numpy.ndarray

        Important layers:
            fm_array[0]  = X stage position
            fm_array[1]  = Y stage position
            fm_array[2]  = DAPI in-focus Z
            fm_array[12] = tissue ID

        fm_array[12] == 0 means filler / do not image.
    """

    # ---------------------------------------------------------
    # Hard-coded microscope geometry
    # ---------------------------------------------------------

    TISSUE_ID_LAYER = 12
    DAPI_Z_LAYER = 2

    x_pixels = 2960
    y_pixels = 2960
    um_per_pixel = 0.204

    overlap = 0.10

    x_step = x_pixels * um_per_pixel * (1 - overlap)
    y_step = y_pixels * um_per_pixel * (1 - overlap)

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------

    if len(xyz_points) == 0:
        raise ValueError(
            "xyz_points must contain at least one (X, Y, Z) point."
        )

    for point in xyz_points:
        if len(point) != 3:
            raise ValueError(
                "Each point must contain exactly (X, Y, Z)."
            )

    # ---------------------------------------------------------
    # Sort tissue centers spatially
    #
    # Highest Y first = top
    # Lowest X first when Y ordering is equal
    # ---------------------------------------------------------

    xyz_points = sorted(
        xyz_points,
        key=lambda point: (-point[1], point[0])
    )

    number_tissues = len(xyz_points)

    # ---------------------------------------------------------
    # Determine rectangular logical layout
    # ---------------------------------------------------------

    block_columns = math.ceil(math.sqrt(number_tissues))
    block_rows = math.ceil(number_tissues / block_columns)

    x_tiles = block_columns * 3
    y_tiles = block_rows * 3

    fm_array = np.zeros(
        (13, y_tiles, x_tiles),
        dtype=np.float64
    )

    # ---------------------------------------------------------
    # Generate each tissue block
    # ---------------------------------------------------------

    for tissue_index, (center_x, center_y, dapi_z) in enumerate(xyz_points):

        block_y = tissue_index // block_columns
        block_x = tissue_index % block_columns

        y_start = block_y * 3
        x_start = block_x * 3

        # -----------------------------------------------------
        # X grows as we move RIGHT
        # -----------------------------------------------------

        x_positions = [
            center_x - x_step,
            center_x,
            center_x + x_step
        ]

        # -----------------------------------------------------
        # Y DECREASES as we move DOWN
        #
        # Array row 0 = highest physical Y
        # Array row 2 = lowest physical Y
        # -----------------------------------------------------

        y_positions = [
            center_y + y_step,
            center_y,
            center_y - y_step
        ]

        # -----------------------------------------------------
        # Tissue ID
        # -----------------------------------------------------

        seed_number = tissue_index + 1

        tissue_fm_code_number = float(
            '1e+' + str(seed_number)
        )

        # -----------------------------------------------------
        # Populate 3x3 block
        # -----------------------------------------------------

        for local_y in range(3):

            for local_x in range(3):
                array_y = y_start + local_y
                array_x = x_start + local_x

                fm_array[
                    0,
                    array_y,
                    array_x
                ] = x_positions[local_x]

                fm_array[
                    1,
                    array_y,
                    array_x
                ] = y_positions[local_y]

                fm_array[
                    DAPI_Z_LAYER,
                    array_y,
                    array_x
                ] = dapi_z

                fm_array[
                    TISSUE_ID_LAYER,
                    array_y,
                    array_x
                ] = tissue_fm_code_number

    return fm_array





points = [(34,56,100), (-1750,2000,130),(-3750,500,200)]
fm = generate_fm_array_from_xyz(points)


print(fm[12])
