import get_datasets

if __name__ == "__main__":
    #get_datasets.generate_modelnet40_dataset(sample_rate=0.25)
    get_datasets.generate_orbit5k_dataset(n_points=1000)
    #get_datasets.generate_obayashi_hiraoka_dataset(bins=32)
    #get_datasets.generate_UIUC_dataset(sub_images=10)