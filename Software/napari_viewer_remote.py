import napari
import s3fs
import zarr
import dask.array as da

endpoint = "http://192.168.1.165:9000"
bucket = "zarr"
dataset = "21_4_26_casey_TMA_6_cluster_1_stitched_final.ome.zarr"

fs = s3fs.S3FileSystem(
    key="admin",
    secret="password123",
    client_kwargs={"endpoint_url": endpoint},
    config_kwargs={"s3": {"addressing_style": "path"}},
)

root_store = fs.get_mapper(f"{bucket}/{dataset}")
root = zarr.open_group(root_store, mode="r")

datasets = root.attrs["multiscales"][0]["datasets"]
paths = [d["path"] for d in datasets]

channels = [
    c.get("label", f"ch{i}")
    for i, c in enumerate(root.attrs["omero"]["channels"])
]

pyramid = []

for path in paths:
    level_store = fs.get_mapper(f"{bucket}/{dataset}/{path}")
    pyramid.append(da.from_zarr(level_store))

viewer = napari.Viewer()

viewer.add_image(
    pyramid,
    channel_axis=0,
    name=channels,
    contrast_limits=[(0, 65535)] * len(channels),
)

napari.run()