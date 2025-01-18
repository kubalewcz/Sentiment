from datasets import load_dataset, concatenate_datasets


dataset1 = load_dataset("clarin-pl/polemo2-official", "all_sentence")
dataset2 = load_dataset("clarin-pl/polemo2-official", "all_text")

merged_dataset_train = concatenate_datasets([dataset1['train'], dataset2['train']])
merged_dataset_val = concatenate_datasets([dataset1['validation'], dataset2['validation']])
merged_dataset_test = concatenate_datasets([dataset1['test'], dataset2['test']])

merged_dataset_train.save_to_disk('./dataset_train')
merged_dataset_test.save_to_disk('./dataset_test')
merged_dataset_val.save_to_disk('./dataset_val')