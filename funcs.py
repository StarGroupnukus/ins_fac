

def is_nose_inside(face):
    # Получаем ключевые точки
    kps = face['kps']
    x_left_eye, x_right_eye = kps[0][0], kps[1][0]
    # Получаем координаты точки носа
    x_nose, y_nose = kps[2]
    check = x_left_eye < x_nose < x_right_eye
    return check
