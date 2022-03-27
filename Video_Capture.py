import face_recognition
import os
import cv2
import matplotlib.image as mpimg
import math
import pickle
from sklearn import neighbors
from face_recognition.face_recognition_cli import image_files_in_folder
from Modules.Capture import FPS
from Modules.Capture import WebcamVideoStream

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

def Show_Locations(face_locations = [], Stream = None, displayed = []):
    
    if Stream is None:
        raise Exception("Stream Not Working")
        
    for name ,(t, r, b, l), acc in face_locations:
        if name not in displayed:
            displayed.append(name)
            print(f'{name} is at location ({(l+r)//2},{(b+t)//2})',end = '\n')
        acc = str(acc) + '%'
        cv2.circle(Stream, ((l+r)//2, (b+t)//2), 150, (255,255,255))
        cv2.putText(Stream, name, (l+6, b-6),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
        cv2.putText(Stream, acc, (r+6, b-6),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
                
    cv2.imshow("Frame", Stream)
    cv2.waitKey(1)
    return displayed

def Video_predict(knn_clf = None, model_path = None, distance_threshold = 0.5, verbose = False):
    
    if knn_clf is None and model_path is None:
        
        print("Training classifier")
        model_path = 'trained_knn_model.clf'
        Classifier = Train("Data/Train", model_save_path = model_path, n_neighbors = 3)
        print("Training Complete")
    
    if knn_clf is None:
        
        with open(model_path, 'rb') as f:
            Classifier = pickle.load(f)
            print("Loading the Model Complete")
            print("To Stop Press Ctrl+C")
    
    attendence = []
    displayed = []
    accuracy = []
    new = []

    try:
        
        vs = WebcamVideoStream(src = 0).start()
        fps = FPS().start()
        while True:
            
            Stream = vs.read()
            if Stream is None:
                raise Exception("Stream Not Working")
        
            Stream_face_locations = face_recognition.face_locations(Stream)
            faces_encodings = face_recognition.face_encodings(Stream, known_face_locations= Stream_face_locations)
            
            if len(faces_encodings) == 0:
                continue
            
            closest_distances = Classifier.kneighbors(faces_encodings, n_neighbors=1)
            
            are_matches = []
            for i in range(len(Stream_face_locations)):
                are_matches.append(closest_distances[0][i][0] <= distance_threshold)
            
            face_locations = []
            for pred, prob, loc, rec in zip(Classifier.predict(faces_encodings), max(Classifier.predict_proba(faces_encodings)), Stream_face_locations, are_matches):
                if rec:
                    if pred not in attendence:
                        attendence.append((pred))
                        accuracy.append(prob*100)
                    face_locations.append((pred, loc, prob*100))
                else:
                    face_locations.append(("unknown", loc))
                    new.append((Stream, loc))
            if verbose:
                displayed = Show_Locations(face_locations, Stream, displayed)
            
    except(KeyboardInterrupt):
        
        if len(attendence) != 0:
            print("Attended People")
            for name, acc in zip(attendence, accuracy):
                print(f"{name} and I'm {acc}% sure.", end = '\n')
                
        if len(new) != 0:
            print("New People")
            i = 2
            for img, (t,r,b,l) in new:
                path = './Data/Noobie/unknown_' + str(i) + '.jpg'
                cv2.imwrite(path,img)
                print(f"({t},{r},{b},{l}) is location of the noobie")
                i += 1
        
        return 'Thank you, Bye'
    
    finally:
        fps.update()
        fps.stop()
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    msg = Video_predict(model_path = './Model/trained_knn_model.clf', distance_threshold = 0.6, verbose = False)
    print(msg)
    
    
