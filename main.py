
import insightface
from insightface.app import FaceAnalysis
import cv2
import timeit
import os

detector = insightface.model_zoo.get_model('models/w600k_mbf.onnx', download=True)
app = FaceAnalysis()
app.prepare(ctx_id=0)


def compare_images(image1, image2):
    try:
        img1 = cv2.imread(image1)
        img2 = cv2.imread(image2)

        face_1 = app.get(img1)[0]['embedding']
        face_2 = app.get(img2)[0]['embedding']
        score = detector.compute_sim(face_1, face_2)
        print(image1)
        return score * 100
    except Exception as e:
        print(e, image1)
        return -1
#old func
def compare_all(image):
    folder = './images'
    best_match = []
    for img in os.listdir(folder):
        res = compare_images(image, f'{folder}/{img}')
        best_match.append((res, img))
    return max(best_match)


#new func
def compare_with_folders(input_image, base_folder):
    best_match = None
    max_score = -1

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            match_count = 0
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                score = compare_images(input_image, img_path)
                print(score, folder_name)
                if score > max_score:
                    max_score = score
                    best_match = folder_name
                # Если требуется задать порог совпадения, можно добавить условие, например:
                # if score >= threshold:
                #     match_count += 1
            # Учитываем порог совпадения:
            # if match_count > max_match_count:
            #     max_match_count = match_count
            #     best_match = folder_name

    return best_match


verify = 'test_images/20_2024-02-26-20-24-35.jpg'
path_images = 'test_images'
start = timeit.default_timer()
for test_image in os.listdir(path_images):

    verified = compare_with_folders(os.path.join(path_images, test_image), 'images')
stop = timeit.default_timer()

print(verified)
# path = f'./images/{verified}'


# def attendance_face(path):
#     try:
#         with open('db.json', 'r') as file:
#             data = json.load(file)
#         for person in data:
#             if person['name'] in path:
#                 person["attendance"] = True
#         with open('db.json', 'w') as file:
#             json.dump(data, file, indent=2)
#     except IOError as error:
#         print("File input error:", error)
#     except TypeError as error:
#         print("Type error:", error)
#
#
# attendance_face(path)

print(f'time: {stop - start} seconds')
