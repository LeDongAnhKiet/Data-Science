from imutils import paths
import random, shutil, os, config

original_paths = list(paths.list_images(config.INPUT_DATASET))
random.seed(7)
random.shuffle(original_paths)

index = int(len(original_paths) * config.TRAIN_SPLIT)
train_paths = original_paths[:index]
test_paths = original_paths[index:]

index = int(len(train_paths) * config.VAL_SPLIT)
val_paths = train_paths[:index]
train_paths = train_paths[index:]

datasets = [('training', train_paths, config.TRAIN_PATH),
            ('validation', val_paths, config.VAL_PATH),
            ('testing', test_paths, config.TEST_PATH)]

for set_type, original_paths, base_path in datasets:
    print(f'Building {set_type} set')
    # Create base path if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    for path in original_paths:
        file_name = os.path.basename(path)
        label = file_name[-5:-4]
        label_path = os.path.join(base_path, label)
        os.makedirs(label_path, exist_ok=True)
        new_path = os.path.join(label_path, file_name)
        shutil.copy2(path, new_path)