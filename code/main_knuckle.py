# main_knuckle.py - Complete Knuckle Biometric System
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from collections import Counter

print("🚀 Starting Knuckle Biometric System...")

class KnuckleSystem:
    def __init__(self):
        self.model = None
        self.encoder = LabelEncoder()
        
    def load_dataset(self):
        print("📁 Loading dataset...")
        images, labels = [], []
        
        for i in range(1, 12):  # person1 to person11
            folder = f"person{i}"
            path = f"dataset/{folder}"
            if os.path.exists(path):
                count = 0
                for file in os.listdir(path):
                    if file.endswith('.bmp'):
                        img_path = f"{path}/{file}"
                        img = cv2.imread(img_path, 0)  # grayscale
                        if img is not None:
                            img = cv2.resize(img, (50, 50))
                            img = img.astype('float32') / 255.0
                            images.append(img)
                            labels.append(folder)
                            count += 1
                status = "✅" if count > 0 else "⚠️ "
                print(f"{status} {folder}: {count} images")
        
        print(f"📊 Total: {len(images)} images")
        return images, labels
    
    def filter_classes(self, images, labels, min_samples=2):
        counts = Counter(labels)
        filtered_img, filtered_lbl = [], []
        removed = []
        
        for img, lbl in zip(images, labels):
            if counts[lbl] >= min_samples:
                filtered_img.append(img)
                filtered_lbl.append(lbl)
            elif lbl not in removed:
                removed.append(lbl)
                print(f"⚠️  Removing {lbl} (only {counts[lbl]} image)")
        
        print(f"✅ After filter: {len(filtered_img)} images")
        return filtered_img, filtered_lbl
    
    def extract_features(self, images):
        print("🔧 Extracting features...")
        features = []
        
        for i, img in enumerate(images):
            img_uint8 = (img * 255).astype('uint8')
            feat = []
            
            # Raw pixels
            feat.extend(img_uint8.flatten())
            
            # Basic stats
            feat.extend([np.mean(img_uint8), np.std(img_uint8)])
            
            # Histogram
            hist = cv2.calcHist([img_uint8], [0], None, [8], [0, 256])
            feat.extend(hist.flatten())
            
            features.append(feat)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(images)}")
        
        print(f"📐 Features per image: {len(features[0])}")
        return np.array(features)
    
    def train(self):
        print("\n🤖 Training model...")
        images, labels = self.load_dataset()
        
        if len(images) < 10:
            print("❌ Not enough images!")
            return 0
        
        images, labels = self.filter_classes(images, labels, 2)
        
        X = self.extract_features(images)
        y = self.encoder.fit_transform(labels)
        
        print(f"🎯 Training on {X.shape[0]} samples")
        print(f"👥 Persons: {list(self.encoder.classes_)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_train, y_train)
        
        accuracy = self.model.score(X_test, y_test)
        print(f"📈 ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy
    
    def test_predictions(self):
        if not self.model:
            print("❌ Model not trained!")
            return 0
        
        print("\n🧪 Testing predictions...")
        results = []
        
        for person in self.encoder.classes_:
            test_img = f"dataset/{person}/1.1.bmp"
            if os.path.exists(test_img):
                img = cv2.imread(test_img, 0)
                if img is not None:
                    img = cv2.resize(img, (50, 50))
                    img = img.astype('float32') / 255.0
                    features = self.extract_features([img])
                    pred_idx = self.model.predict(features)[0]
                    pred_person = self.encoder.inverse_transform([pred_idx])[0]
                    correct = (pred_person == person)
                    results.append(correct)
                    
                    status = "" if correct else "❌"
                    conf = np.max(self.model.predict_proba(features))
                    print(f"{status} {person} -> {pred_person} (conf: {conf:.3f})")
        
        if results:
            accuracy = np.mean(results)
            print(f" TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f" Correct: {sum(results)}/{len(results)}")
            return accuracy
        return 0
    
    def save_model(self):
        if self.model:
            joblib.dump({'model': self.model, 'encoder': self.encoder}, 'knuckle_model.pkl')
            print(" Model saved as 'knuckle_model.pkl'")

def main():
    print("=" * 50)
    print(" KNUCKLE BIOMETRIC SYSTEM")
    print("=" * 50)
    
    system = KnuckleSystem()
    accuracy = system.train()
    
    if accuracy > 0:
        system.test_predictions()
        system.save_model()
        print("\n SYSTEM READY!")
    else:
        print("\n Training failed!")

if __name__ == "__main__":
    main()