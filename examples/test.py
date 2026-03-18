from geo_inference.geo_inference import GeoInference

def main() -> None:
    geo_inference = GeoInference(
        model="/export/sata01/wspace/test_dir/multi/demo/pt2/dofav2_67_0.418.pt2",
        work_dir="/export/sata01/wspace/test_dir/multi/demo/prediction",
        device="gpu",
    )

    image_path = "/export/sata01/wspace/test_dir/multi/demo/image/ns2_small_rgbn.tif"
    geo_inference(
        inference_input=image_path,
        sensor_name="worldview-2-rgbn",
        patch_size=512,
    )


if __name__ == "__main__":
    main()
