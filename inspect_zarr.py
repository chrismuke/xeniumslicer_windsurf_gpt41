import zarr
import sys

def inspect_zarr(zarr_path):
    z = zarr.open(zarr_path, mode="r")
    print(f"Array keys: {list(z.array_keys())}")
    print(f"Group keys: {list(z.group_keys())}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path", type=str, help="Path to Zarr group")
    args = parser.parse_args()
    inspect_zarr(args.zarr_path)
