from datasets import load_from_disk


dataset_path = 'dataset/dataset_train'

ds = load_from_disk(dataset_path)

print(ds)