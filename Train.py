import math
from sklearn import neighbors
import os
import os.path
import pickle
import matplotlib.image as mpimg
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

def Train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose= False):
    images = []
    labels = []
    
    for class_dir in os.listdir(train_dir):
        
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            
            img = mpimg.imread(img_path)
            face_bounding_boxes = face_recognition.face_locations(img)
            
            if len(face_bounding_boxes) != 1:
                if verbose:
                    if len(face_bounding_boxes) < 1:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face"))
                    else:
                        print("Image {} not suitable for training: {}".format(img_path, "Found more than one face"))
            else:
                images.append(face_recognition.face_encodings(img, known_face_locations = face_bounding_boxes)[0])
                labels.append(class_dir)
                
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(images))))
        
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
            
    Classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    Classifier.fit(images, labels)
    
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(Classifier, f)
            
    return Classifier


if __name__ == "__main__":
    print("Training KNN classifier...")
    Train("Data/Train", model_save_path="./Model/trained_knn_model.clf", verbose = True)
    print("Training complete!")