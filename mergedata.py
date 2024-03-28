
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def image_paths_helper(folder_path):
    image_paths = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                image_paths.append(image_path)
    return image_paths

def create_dataset(path):
    dataset = []
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(path):
            image_paths = image_paths_helper(folder_path)
            dataset.extend(image_paths)
    return np.array(dataset)

def create_dataset2(path):
    dataset_N = []
    dataset_O_and_R = []
   
    for folder_name in os.listdir(path):
       
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            image_paths = image_paths_helper(folder_path)
            if folder_name == 'N':
                dataset_N.extend(image_paths)
            elif folder_name in ['O', 'R']:
                dataset_O_and_R.extend(image_paths)
    
    dataset_O_and_R = np.array(dataset_O_and_R) 
    return np.array(dataset_N), dataset_O_and_R


data_folder = 'Data/KaggleCEN/data'
data_folder2 = 'Data/KaggleWang'
data_sap_test = 'Data/KaggleSAP/DATASET/TEST'
data_sap_train = 'Data/KaggleSAP/DATASET/TRAIN'
kaggleCenData = create_dataset(data_folder)
kaggleWangData = create_dataset(data_folder2)
merged_dataset1 = np.concatenate((kaggleCenData , kaggleWangData))
dataset_N, dataset_O_and_R = create_dataset2(data_sap_test)
dataset_N2, dataset_O_and_R2 = create_dataset2(data_sap_train)
all_dataset_N = np.concatenate((dataset_N , dataset_N2))
all_dataset_O_and_R = np.concatenate((dataset_O_and_R, dataset_O_and_R2 ))
waste_data = np.concatenate((merged_dataset1, all_dataset_O_and_R))


label_encoder = LabelEncoder()
waste_data_labels = np.array([os.path.basename(os.path.dirname(path)) for path in waste_data])
waste_data_labels_encoded = label_encoder.fit_transform(waste_data_labels)

train_images_waste_data, test_images_waste_data, train_labels_waste_data, test_labels_waste_data = train_test_split(waste_data, waste_data_labels_encoded, test_size=0.2, random_state=42)



print("Number of training images in waste_data:", len(train_images_waste_data))
print("Number of testing images in waste_data:", len(test_images_waste_data))









