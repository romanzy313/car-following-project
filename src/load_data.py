import zarr


# need to load zar, and to extract it
def load_data(path: str):
    data: zarr.Group = zarr.open(path, mode="r")  # type: ignore
    print("given data info is", data.info)
