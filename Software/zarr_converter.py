import math
import pandas as pd
import tifffile
import dask.array as da
import zarr
import os
from ome_zarr.writer import write_multiscale

csv_path = r"C:\Users\anderson\Downloads\CMPA_multiplex_panels_history - full_panel_4_21_26.csv"

os.chdir(r'C:\Users\anderson\Documents\images')
input_path = "21_4_26_casey_TMA_6_cluster_1_stitched.ome.tif"
output_path = "21_4_26_casey_TMA_6_cluster_1_stitched_final.ome.zarr"


def clean_name(x):
    if x is None:
        return "nothing"
    if isinstance(x, float) and math.isnan(x):
        return "nothing"
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return "nothing"
    return x


# ---------- BUILD CHANNEL NAMES ----------
channel_names = [
    "DAPI",
    "autoF_488",
    "autoF_555",
    "autoF_647",
    "autoF_750",
]

df = pd.read_csv(csv_path)

raw_layout = df[
    ["Unnamed: 17", "Unnamed: 18", "Unnamed: 19", "Unnamed: 20", "Unnamed: 21"]
].copy()

raw_layout.columns = ["round", "DAPI", "488", "555", "647"]

started = False

for _, row in raw_layout.iterrows():
    round_value = clean_name(row["round"])

    if not started:
        if round_value.lower() == "round":
            started = True
        continue

    if round_value == "nothing":
        break

    if round_value == "1 secondary":
        continue

    channel_names.append(clean_name(row["DAPI"]))
    channel_names.append(clean_name(row["488"]))
    channel_names.append(clean_name(row["555"]))
    channel_names.append(clean_name(row["647"]))
    channel_names.append("nothing")


print(f"Generated {len(channel_names)} channel names:")
for i, name in enumerate(channel_names):
    print(i, name)


# ---------- READ TIFF PYRAMID ----------
with tifffile.TiffFile(input_path) as tif:
    series = tif.series[0]
    pyramid = []

    print(f"Number of pyramid levels found: {len(series.levels)}")

    for i, level in enumerate(series.levels):
        arr = level.asarray()
        print(f"Level {i}: shape={arr.shape}, dtype={arr.dtype}")

        if i == 0 and arr.shape[0] != len(channel_names):
            raise ValueError(
                f"Channel mismatch: TIFF has {arr.shape[0]} channels, "
                f"but sheet generated {len(channel_names)} names."
            )

        pyramid.append(
            da.from_array(arr, chunks=(1, 1024, 1024))
        )


# ---------- WRITE ZARR V2 ----------
try:
    root = zarr.open_group(
        output_path,
        mode="w",
        zarr_format=2
    )
except TypeError:
    root = zarr.open_group(
        output_path,
        mode="w",
        zarr_version=2
    )


write_multiscale(
    pyramid=pyramid,
    group=root,
    axes=[
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ],
    name="SpectrePlex_Image",
)


# ---------- ADD OMERO CHANNEL METADATA ----------
channel_colors = [
    "FFFFFF",  # DAPI
    "00FF00",  # 488
    "FF00FF",  # 555
    "FF0000",  # 647
    "00FFFF",  # 750
]

root.attrs["omero"] = {
    "name": "SpectrePlex_Image",
    "version": "0.4",
    "channels": [
        {
            "label": name,
            "color": channel_colors[i % 5],
            "active": True,
            "coefficient": 1,
            "family": "linear",
            "inverted": False,
            "window": {
                "start": 0,
                "end": 65535,
                "min": 0,
                "max": 65535,
            },
        }
        for i, name in enumerate(channel_names)
    ],
    "rdefs": {
        "defaultT": 0,
        "defaultZ": 0,
        "model": "color",
    },
}


# ---------- VERIFY ----------
attrs = root.attrs.asdict()

print("\nMetadata check:")
print("Has multiscales:", "multiscales" in attrs)
print("Has omero:", "omero" in attrs)
print("First channel:", attrs["omero"]["channels"][0])

print("\nDone.")
print(f"OME-Zarr written to: {output_path}")