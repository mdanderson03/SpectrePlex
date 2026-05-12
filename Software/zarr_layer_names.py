import tifffile
import dask.array as da
import zarr

from ome_zarr.writer import write_multiscale


input_path = r"C:\path\to\your\input.ome.tif"
output_path = r"C:\path\to\your\output.ome.zarr"

# Your shape is (60, Y, X), so this is probably channels, Y, X
axes = "cyx"

with tifffile.TiffFile(input_path) as tif:
    series = tif.series[0]

    print(f"Number of pyramid levels found: {len(series.levels)}")

    pyramid = []

    for i, level in enumerate(series.levels):
        arr = level.asarray()
        print(f"Level {i}: shape={arr.shape}, dtype={arr.dtype}")

        dask_arr = da.from_array(
            arr,
            chunks=(1, 1024, 1024)
        )

        pyramid.append(dask_arr)


root = zarr.open_group(output_path, mode="w")

write_multiscale(
    pyramid=pyramid,
    group=root,
    axes=axes,
    name="SpectrePlex_Image",
)

print("Done.")
print(f"OME-Zarr written to: {output_path}")