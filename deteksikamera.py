import cv2

def resize_with_aspect_ratio(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Buka kamera
video = cv2.VideoCapture(0)

# Atur resolusi kamera
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Buat jendela bisa di-*maximize*
cv2.namedWindow('Deteksi Wajah Langsung', cv2.WINDOW_NORMAL)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontal (hilangkan efek mirror)

    # Deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Gambar kotak wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tampilkan jumlah wajah
    cv2.putText(frame, f'Jumlah Wajah Terdeteksi: {len(faces)}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Resize hasil tampilan (misal max width 960)
    tampil = resize_with_aspect_ratio(frame, width=960)

    # Tampilkan
    cv2.imshow('Deteksi Wajah Langsung', tampil)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
