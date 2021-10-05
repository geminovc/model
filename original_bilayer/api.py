import face_alignment

def extract_keypoints(input_img):
    # Stickman/facemasks drawer
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
    return fa.get_landmarks(input_img)