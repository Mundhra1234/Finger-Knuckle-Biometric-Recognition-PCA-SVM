
import os

print(" Checking your dataset structure...")
print("Current directory:", os.getcwd())
print("Files here:", os.listdir('.'))

if os.path.exists('dataset'):
    print("\n Dataset folder found!")
    persons = os.listdir('dataset')
    print(f"Person folders: {persons}")
    
    for person in persons:
        person_path = os.path.join('dataset', person)
        if os.path.isdir(person_path):
            images = [f for f in os.listdir(person_path) if f.endswith('.bmp')]
            print(f"  {person}: {len(images)} images")
else:
    print("\n No dataset folder found!")

print("\nNext: Run 'python setup_knuckle.py'")