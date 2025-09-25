import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment
import time

# ---------------- CONFIG ----------------
YOLO_MODEL_PATH = r"C:\Users\Aditri\Desktop\Drone\runs\detect\train4\weights\best.onnx"
OSNET_MODEL_PATH = r"C:\Users\Aditri\Desktop\Drone\yolov5_tracking\weights\osnet_x0_25.onnx"
VIDEO_PATH = r"C:\Users\Aditri\Desktop\Drone\videoplayback.mp4"
INPUT_SIZE = 640
CONF_THRESH = 0.29
IOU_THRESH = 0.45
TRACK_MAX_AGE = 5
CLASS_ID_UAV = 1  # Only UAVs

# ---------------- Letterbox ----------------
def letterbox(img, new_shape=INPUT_SIZE, color=(114,114,114)):
    shape = img.shape[:2]  # h, w
    r = min(new_shape/shape[0], new_shape/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw //= 2; dh //= 2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, dw, dh

# ---------------- NMS ----------------
def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    area1 = (box[2]-box[0])*(box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-6)

def nms(boxes, scores, iou_thresh=0.5):
    if len(boxes)==0: return []
    idxs = scores.argsort()[::-1]
    keep = []
    boxes = np.array(boxes)
    while len(idxs)>0:
        i = idxs[0]; keep.append(i)
        if len(idxs)==1: break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious<iou_thresh]
    return keep

# ---------------- Preprocess YOLO ----------------
def preprocess_yolo(img):
    img, r, dw, dh = letterbox(img, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis=0)
    return img, r, dw, dh

# ---------------- YOLO Postprocess ----------------
def postprocess_yolo(outputs, img_shape, ratio, dw, dh):
    preds = outputs[0]
    preds = np.squeeze(preds).T
    boxes = preds[:,:4]
    scores = preds[:,4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]

    # Filter only UAVs
    mask = (confidences > CONF_THRESH) & (class_ids == CLASS_ID_UAV)
    boxes = boxes[mask]; confidences = confidences[mask]; class_ids = class_ids[mask]

    if len(boxes)==0: return [], [], []

    # xywh → xyxy
    x = boxes[:,0]; y = boxes[:,1]; w = boxes[:,2]; h = boxes[:,3]
    x1 = (x - w/2 - dw)/ratio
    y1 = (y - h/2 - dh)/ratio
    x2 = (x + w/2 - dw)/ratio
    y2 = (y + h/2 - dh)/ratio
    boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1)

    h_img, w_img = img_shape
    boxes_xyxy[:,[0,2]] = np.clip(boxes_xyxy[:,[0,2]],0,w_img-1)
    boxes_xyxy[:,[1,3]] = np.clip(boxes_xyxy[:,[1,3]],0,h_img-1)
    keep = nms(boxes_xyxy, confidences, IOU_THRESH)
    return boxes_xyxy[keep], confidences[keep], class_ids[keep]

# ---------------- OSNet Embeddings ----------------
def get_embedding(crop, session, input_name):
    crop = cv2.resize(crop,(128,256))
    crop = crop.astype(np.float32)/255.0
    crop = np.transpose(crop,(2,0,1))
    crop = np.expand_dims(crop, axis=0)
    feat = session.run(None, {input_name: crop})[0]
    feat = feat/np.linalg.norm(feat,axis=1,keepdims=True)
    return feat[0]

# ---------------- Kalman + Tracker ----------------
class KalmanTrack:
    def __init__(self, bbox, embedding, track_id):
        self.bbox = np.array(bbox,dtype=np.float32)
        self.embedding = embedding
        self.track_id = track_id
        self.age = 0
        self.kf = self.init_kalman(bbox)

    def init_kalman(self, bbox):
        kf = cv2.KalmanFilter(8,4)
        kf.measurementMatrix = np.eye(4,8,dtype=np.float32)
        kf.transitionMatrix = np.eye(8,dtype=np.float32)
        kf.statePre[:4,0] = bbox.reshape(4)
        return kf

    def predict(self):
        pred = self.kf.predict()
        self.bbox = pred[:4,0]
        return self.bbox

    def update(self,bbox,embedding):
        measurement = np.array(bbox,dtype=np.float32).reshape(4,1)
        self.kf.correct(measurement)
        self.bbox = bbox
        self.embedding = embedding
        self.age = 0

class Tracker:
    def __init__(self,max_age=5,emb_thresh=0.5):
        self.tracks = []
        self.next_id = 0
        self.max_age = max_age
        self.emb_thresh = emb_thresh

    def update(self,detections,embeddings):
        # Predict all tracks
        for tr in self.tracks:
            tr.predict()

        if len(detections)==0:
            for t in self.tracks: t.age+=1
            self.tracks = [t for t in self.tracks if t.age<self.max_age]
            return self.tracks

        if len(self.tracks)==0:
            for det,emb in zip(detections,embeddings):
                self.tracks.append(KalmanTrack(det,emb,self.next_id))
                self.next_id+=1
            return self.tracks

        # Cost matrix: 1-IoU + embedding dist
        cost_matrix = np.zeros((len(self.tracks),len(detections)),dtype=np.float32)
        for i,tr in enumerate(self.tracks):
            for j,det in enumerate(detections):
                iou = compute_iou(np.array(tr.bbox),np.array([det]))[0]
                dist = np.linalg.norm(tr.embedding - embeddings[j])
                cost_matrix[i,j] = (1-iou) + dist

        # Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks, assigned_dets = [],[]
        for r,c in zip(row_ind,col_ind):
            if cost_matrix[r,c] < 1+self.emb_thresh:
                self.tracks[r].update(detections[c], embeddings[c])
                assigned_tracks.append(r)
                assigned_dets.append(c)

        # Add new unmatched detections
        for i,det in enumerate(detections):
            if i not in assigned_dets:
                self.tracks.append(KalmanTrack(det,embeddings[i],self.next_id))
                self.next_id+=1

        # Age unmatched tracks
        for i,t in enumerate(self.tracks):
            if i not in assigned_tracks: t.age+=1
        self.tracks = [t for t in self.tracks if t.age<self.max_age]

        return self.tracks

# ---------------- MAIN ----------------
def main():
    yolo_session = ort.InferenceSession(YOLO_MODEL_PATH, providers=['CPUExecutionProvider'])
    yolo_input = yolo_session.get_inputs()[0].name
    osnet_session = ort.InferenceSession(OSNET_MODEL_PATH, providers=['CPUExecutionProvider'])
    osnet_input = osnet_session.get_inputs()[0].name

    tracker = Tracker(max_age=TRACK_MAX_AGE, emb_thresh=0.5)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: cannot open video.")
        return

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break

        img_input, ratio, dw, dh = preprocess_yolo(frame)
        outputs = yolo_session.run(None,{yolo_input: img_input})
        boxes, scores, class_ids = postprocess_yolo(outputs, frame.shape[:2], ratio, dw, dh)

        embeddings=[]
        for box in boxes:
            x1,y1,x2,y2 = map(int, box)
            crop = frame[y1:y2,x1:x2]
            if crop.size==0: continue
            emb = get_embedding(crop, osnet_session, osnet_input)
            embeddings.append(emb)

        tracks = tracker.update(boxes, embeddings)

        # Draw tracks
        for t in tracks:
            x1,y1,x2,y2 = map(int,t.bbox)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID:{t.track_id}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        # FPS
        curr_time = time.time()
        fps = 1/(curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame,f"FPS:{fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.imshow("UAV Tracker", frame)
        if cv2.waitKey(1)==27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
