from ultralytics import YOLO
import cv2
import os

# === Configurações iniciais ===
model_path = "best_nano.pt"
if not os.path.exists(model_path):
    print(f"Modelo não encontrado: {model_path}")
    exit()

model = YOLO(model_path)

bad_classes = {"bad apple", "bad banana", "bad orange", "bad pomegranate"}
good_classes = {"good apple", "good banana", "good orange", "good pomegranate"}

def detectar_frutas_com_slider_video(video_path, modo="ruins"):
    if not os.path.exists(video_path):
        print(f"[ERRO] Vídeo não encontrado: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {video_path}")
        return

    names = model.names
    if modo == "boas":
        classes_filtradas = good_classes
        cor = (0, 255, 0)
    else:
        classes_filtradas = bad_classes
        cor = (0, 0, 255)

    window_name = "Detecção"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.createTrackbar("Confiança (%)", window_name, 50, 100, lambda x: None)
    cv2.createTrackbar("Máx. Itens", window_name, 10, 20, lambda x: None)
    cv2.waitKey(1)

    print("[INFO] Pressione ESC para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fim do vídeo ou erro ao capturar frame.")
            break

        results = model(frame)
        result = results[0]

        min_conf = cv2.getTrackbarPos("Confiança (%)", window_name) / 100.0
        max_items = cv2.getTrackbarPos("Máx. Itens", window_name)
        if max_items < 1:
            max_items = 1

        caixas_filtradas = []
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < min_conf:
                continue
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            if cls_name in classes_filtradas:
                caixas_filtradas.append((conf, cls_name, box))

        caixas_filtradas.sort(reverse=True, key=lambda x: x[0])
        caixas_mostradas = caixas_filtradas[:max_items]

        img = frame.copy()
        for conf, cls_name, box in caixas_mostradas:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        cv2.imshow(window_name, img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "laranjas.mp4"
    detectar_frutas_com_slider_video(video_path, modo="boas")
