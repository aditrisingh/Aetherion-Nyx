#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <fstream>

using namespace std;
using namespace nvinfer1;

// ---------------- CONFIG ----------------
const string YOLO_ENGINE_PATH = "/mnt/c/Users/Aditri/Desktop/Drone/yolofp16.engine";
const string OSNET_ENGINE_PATH = "/mnt/c/Users/Aditri/Desktop/Drone/osnet_fp16.engine";
const string VIDEO_PATH = "/mnt/c/Users/Aditri/Desktop/Drone/videoplayback.mp4";
const bool USE_WEBCAM = false; // Set to true for webcam, false for video file

const int INPUT_SIZE = 640;
const float CONF_THRESH = 0.29f;
const float IOU_THRESH = 0.45f;
const int TRACK_MAX_AGE = 5;
const float EMB_THRESH = 0.5f;
const int CLASS_ID_UAV = 1; // Only UAVs

// ---------------- TensorRT Logger ----------------
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            cout << msg << endl;
        }
    }
};

// ---------------- Load TensorRT Engine ----------------
ICudaEngine* loadEngine(const string& enginePath, IRuntime* runtime) {
    ifstream engineFile(enginePath, ios::binary);
    if (!engineFile) {
        cerr << "Error: cannot open engine file: " << enginePath << endl;
        return nullptr;
    }
    engineFile.seekg(0, ios::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, ios::beg);
    vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    return runtime->deserializeCudaEngine(engineData.data(), engineSize);
}

// ---------------- Helpers ----------------
struct Box { float x1, y1, x2, y2; };

vector<float> compute_iou_arr(const Box &box, const vector<Box> &boxes) {
    vector<float> res; res.reserve(boxes.size());
    float area1 = max(0.0f, box.x2 - box.x1) * max(0.0f, box.y2 - box.y1);
    for (const auto &b : boxes) {
        float xx1 = max(box.x1, b.x1);
        float yy1 = max(box.y1, b.y1);
        float xx2 = min(box.x2, b.x2);
        float yy2 = min(box.y2, b.y2);
        float inter = max(0.0f, xx2 - xx1) * max(0.0f, yy2 - yy1);
        float area2 = max(0.0f, b.x2 - b.x1) * max(0.0f, b.y2 - b.y1);
        float uni = area1 + area2 - inter;
        res.push_back(inter / max(1e-6f, uni));
    }
    return res;
}

vector<int> nms_indices(const vector<Box> &boxes, const vector<float> &scores, float iou_thresh) {
    if (boxes.empty()) return {};
    vector<int> idxs(boxes.size()); iota(idxs.begin(), idxs.end(), 0);
    sort(idxs.begin(), idxs.end(), [&](int a, int b){ return scores[a] > scores[b]; });
    vector<int> keep;
    while (!idxs.empty()) {
        int i = idxs[0]; keep.push_back(i);
        if (idxs.size() == 1) break;
        vector<int> remaining;
        for (size_t k = 1; k < idxs.size(); ++k) {
            int j = idxs[k];
            float iou = compute_iou_arr(boxes[i], vector<Box>{boxes[j]})[0];
            if (iou < iou_thresh) remaining.push_back(j);
        }
        idxs = remaining;
    }
    return keep;
}

// ---------------- Letterbox ----------------
cv::Mat letterbox(const cv::Mat &img, int new_shape, float &r, int &dw, int &dh, const cv::Scalar &color=cv::Scalar(114,114,114)) {
    int h = img.rows, w = img.cols;
    r = std::min((float)new_shape / (float)h, (float)new_shape / (float)w);
    int new_unpad_w = (int)round(w * r);
    int new_unpad_h = (int)round(h * r);
    dw = new_shape - new_unpad_w; dh = new_shape - new_unpad_h;
    int dw_half = dw / 2, dh_half = dh / 2;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, dh_half, dh - dh_half, dw_half, dw - dw_half, cv::BORDER_CONSTANT, color);
    dw = dw_half; dh = dh_half;
    return padded;
}

// ---------------- Preprocess YOLO ----------------
void preprocess_yolo(const cv::Mat &frame, vector<float> &out_tensor, float &ratio, int &dw, int &dh) {
    cv::Mat img = letterbox(frame, INPUT_SIZE, ratio, dw, dh);
    cv::Mat rgb; cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::Mat fimg; rgb.convertTo(fimg, CV_32F, 1.0/255.0);
    out_tensor.resize(1 * 3 * INPUT_SIZE * INPUT_SIZE);
    size_t idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < INPUT_SIZE; ++y) {
            for (int x = 0; x < INPUT_SIZE; ++x) {
                out_tensor[idx++] = fimg.at<cv::Vec3f>(y,x)[c];
            }
        }
    }
    cout << "YOLO input tensor shape: [1, 3, " << INPUT_SIZE << ", " << INPUT_SIZE << "]" << endl;
}

// ---------------- Postprocess YOLO ----------------
void postprocess_yolo(const vector<float> &raw, const vector<int64_t> &shape, const cv::Size &img_shape, float ratio, int dw, int dh,
                      vector<Box> &out_boxes, vector<float> &out_scores, vector<int> &out_classids) {
    if (shape.size() < 3) return;
    int64_t dim1 = shape[1], dim2 = shape[2];
    int64_t N = dim1, feat = dim2;
    if (dim2 >= 5 && dim1 < 5 || (dim1 == 1 && dim2 > 5)) {
        // [1, N, feat]
    } else if (dim1 >= 5 && dim2 >= 1 && dim1 < dim2) {
        // [1, feat, N]
        N = dim2; feat = dim1;
    } else {
        if (dim1 >= 5 && dim2 < 5) { feat = dim1; N = dim2; }
    }
    cout << "YOLO output shape: [" << shape[0] << ", " << N << ", " << feat << "]" << endl;

    vector<vector<float>> preds((size_t)N, vector<float>((size_t)feat, 0.0f));
    if ((size_t)(shape[1]) == (size_t)N && (size_t)(shape[2]) == (size_t)feat) {
        for (int64_t i=0;i<N;i++) for (int64_t j=0;j<feat;j++) preds[i][j] = raw[i*feat + j];
    } else {
        for (int64_t i=0;i<N;i++) for (int64_t j=0;j<feat;j++) preds[i][j] = raw[j*N + i];
    }

    vector<Box> boxes_all;
    vector<float> scores_all;
    vector<int> classes_all;

    for (int64_t i = 0; i < N; ++i) {
        if (feat < 5) continue;
        float x = preds[i][0];
        float y = preds[i][1];
        float w = preds[i][2];
        float h = preds[i][3];
        int cstart = 4;
        int classes = feat - cstart;
        if (classes <= 0) continue;
        int best_cls = 0;
        float best_score = preds[i][cstart];
        for (int c = 1; c < classes; ++c) {
            float sc = preds[i][cstart + c];
            if (sc > best_score) { best_score = sc; best_cls = c; }
        }
        if (best_score > CONF_THRESH && best_cls == CLASS_ID_UAV) {
            float x1 = (x - w/2.0f - dw) / ratio;
            float y1 = (y - h/2.0f - dh) / ratio;
            float x2 = (x + w/2.0f - dw) / ratio;
            float y2 = (y + h/2.0f - dh) / ratio;
            x1 = max(0.0f, min(x1, (float)img_shape.width - 1.0f));
            x2 = max(0.0f, min(x2, (float)img_shape.width - 1.0f));
            y1 = max(0.0f, min(y1, (float)img_shape.height - 1.0f));
            y2 = max(0.0f, min(y2, (float)img_shape.height - 1.0f));
            boxes_all.push_back({x1,y1,x2,y2});
            scores_all.push_back(best_score);
            classes_all.push_back(best_cls);
        }
    }

    if (boxes_all.empty()) return;
    vector<int> keep = nms_indices(boxes_all, scores_all, IOU_THRESH);
    for (int idx : keep) {
        out_boxes.push_back(boxes_all[idx]);
        out_scores.push_back(scores_all[idx]);
        out_classids.push_back(classes_all[idx]);
    }
}

// ---------------- OSNet embedding extraction ----------------
vector<float> l2_normalize_vec(const vector<float> &v) {
    double s=0; for(float x: v) s += (double)x*(double)x;
    s = sqrt(max(1e-12, s));
    vector<float> out(v.size());
    for (size_t i=0;i<v.size();++i) out[i] = v[i]/(float)s;
    return out;
}

vector<float> get_embedding(const cv::Mat &crop_bgr, IExecutionContext* context, cudaStream_t stream,
                           const string &input_name, const string &output_name,
                           void* input_gpu, void* output_gpu, int output_size) {
    if (crop_bgr.cols <=0 || crop_bgr.rows <=0) return {};
    cv::Mat crop;
    cv::resize(crop_bgr, crop, cv::Size(128,256));
    cv::Mat rgb; cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);
    cv::Mat fimg; rgb.convertTo(fimg, CV_32F, 1.0/255.0);
    vector<float> input_tensor(1 * 3 * 256 * 128);
    size_t idx = 0;
    for (int c=0;c<3;c++) {
        for (int y=0;y<256;y++) {
            for (int x=0;x<128;x++) {
                input_tensor[idx++] = fimg.at<cv::Vec3f>(y,x)[c];
            }
        }
    }

    cudaMemcpyAsync(input_gpu, input_tensor.data(), input_tensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->setTensorAddress(input_name.c_str(), input_gpu);
    context->setTensorAddress(output_name.c_str(), output_gpu);
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    vector<float> feat(output_size);
    cudaMemcpyAsync(feat.data(), output_gpu, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cout << "OSNet output shape: [" << output_size << "]" << endl;
    return l2_normalize_vec(feat);
}

// ---------------- KalmanTrack & Tracker ----------------
struct KalmanTrack {
    cv::KalmanFilter kf;
    Box bbox;
    vector<float> embedding;
    int track_id;
    int age;
    KalmanTrack(){}
    KalmanTrack(const Box &b, const vector<float> &emb, int id) {
        bbox = b; embedding = emb; track_id = id; age = 0;
        kf = cv::KalmanFilter(8,4,0,CV_32F);
        kf.measurementMatrix = cv::Mat::zeros(4,8,CV_32F);
        for(int i=0;i<4;i++) kf.measurementMatrix.at<float>(i,i) = 1.0f;
        kf.transitionMatrix = cv::Mat::eye(8,8,CV_32F);
        for(int i=0;i<4;i++) kf.statePre.at<float>(i,0) = (&bbox.x1)[i];
    }
    Box predict() {
        cv::Mat pr = kf.predict();
        Box b;
        b.x1 = pr.at<float>(0,0);
        b.y1 = pr.at<float>(1,0);
        b.x2 = pr.at<float>(2,0);
        b.y2 = pr.at<float>(3,0);
        bbox = b;
        return b;
    }
    void update(const Box &b, const vector<float> &emb) {
        cv::Mat meas = cv::Mat::zeros(4,1,CV_32F);
        meas.at<float>(0,0) = b.x1; meas.at<float>(1,0) = b.y1; meas.at<float>(2,0) = b.x2; meas.at<float>(3,0) = b.y2;
        kf.correct(meas);
        bbox = b; embedding = emb; age = 0;
    }
};

vector<pair<int,int>> hungarian_assign(const vector<vector<float>> &cost) {
    int n = cost.size();
    int m = (n>0 ? cost[0].size() : 0);
    int dim = max(n,m);
    if (dim==0) return {};
    const float INF = 1e9f;
    vector<vector<float>> a(dim, vector<float>(dim, INF));
    for (int i=0;i<n;i++) for (int j=0;j<m;j++) a[i][j] = cost[i][j];
    vector<float> u(dim+1,0), v(dim+1,0);
    vector<int> p(dim+1,0), way(dim+1,0);
    for (int i=1;i<=dim;i++) {
        p[0]=i; int j0=0;
        vector<float> minv(dim+1, INF); vector<char> used(dim+1,false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0; float delta = INF;
            for (int j=1;j<=dim;j++) if (!used[j]) {
                float cur = a[i0-1][j-1]-u[i0]-v[j];
                if (cur<minv[j]) { minv[j]=cur; way[j]=j0; }
                if (minv[j]<delta) { delta = minv[j]; j1=j; }
            }
            for (int j=0;j<=dim;j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0]!=0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    vector<int> assign(dim, -1);
    for (int j=1;j<=dim;j++) if (p[j] <= dim) assign[p[j]-1] = j-1;
    vector<pair<int,int>> matches;
    for (int i=0;i<n;i++) {
        int j = assign[i];
        if (j>=0 && j<m && a[i][j] < INF/2) matches.emplace_back(i,j);
    }
    return matches;
}

struct Tracker {
    vector<KalmanTrack> tracks;
    int next_id;
    int max_age;
    float emb_thresh;
    Tracker(int max_age_=5, float emb_thresh_=0.5f) : next_id(0), max_age(max_age_), emb_thresh(emb_thresh_) {}
    vector<KalmanTrack> update(const vector<Box> &detections, const vector<vector<float>> &embeddings) {
        for (auto &t: tracks) t.predict();
        if (detections.empty()) {
            for (auto &t: tracks) t.age++;
            vector<KalmanTrack> alive;
            for (auto &t: tracks) if (t.age < max_age) alive.push_back(t);
            tracks = alive;
            return tracks;
        }
        if (tracks.empty()) {
            for (size_t i=0;i<detections.size();++i) {
                tracks.emplace_back(detections[i], embeddings[i], next_id++);
            }
            return tracks;
        }
        int T = (int)tracks.size(), D = (int)detections.size();
        vector<vector<float>> cost(T, vector<float>(D, 0.0f));
        for (int i=0;i<T;i++) {
            for (int j=0;j<D;j++) {
                float iou = compute_iou_arr(tracks[i].bbox, vector<Box>{detections[j]})[0];
                float dist=0;
                for (size_t k=0;k<tracks[i].embedding.size();++k) {
                    float d = tracks[i].embedding[k] - embeddings[j][k];
                    dist += d*d;
                }
                dist = sqrtf(dist);
                cost[i][j] = (1.0f - iou) + dist;
            }
        }
        auto assigns = hungarian_assign(cost);
        vector<int> assigned_tracks, assigned_dets;
        for (auto &pr : assigns) {
            int r = pr.first, c = pr.second;
            if (cost[r][c] < 1.0f + emb_thresh) {
                tracks[r].update(detections[c], embeddings[c]);
                assigned_tracks.push_back(r);
                assigned_dets.push_back(c);
            }
        }
        for (int i=0;i<D;i++) {
            if (find(assigned_dets.begin(), assigned_dets.end(), i) == assigned_dets.end()) {
                tracks.emplace_back(detections[i], embeddings[i], next_id++);
            }
        }
        for (int i=0;i<(int)tracks.size();++i) {
            if (find(assigned_tracks.begin(), assigned_tracks.end(), i) == assigned_tracks.end()) tracks[i].age++;
        }
        vector<KalmanTrack> alive;
        for (auto &t: tracks) if (t.age < max_age) alive.push_back(t);
        tracks = alive;
        return tracks;
    }
};

// ---------------- MAIN ----------------
int main() {
    try {
        // Initialize CUDA context
        cudaError_t err = cudaFree(0);
        if (err != cudaSuccess) {
            cerr << "CUDA context init failed: " << cudaGetErrorString(err) << endl;
            return -1;
        }
        cout << "CUDA context initialized" << endl;

        // TensorRT setup
        Logger gLogger;
        IRuntime* runtime = createInferRuntime(gLogger);
        if (!runtime) throw runtime_error("Failed to create TensorRT runtime");

        ICudaEngine* yolo_engine = loadEngine(YOLO_ENGINE_PATH, runtime);
        if (!yolo_engine) throw runtime_error("Failed to load YOLO engine");

        ICudaEngine* osnet_engine = loadEngine(OSNET_ENGINE_PATH, runtime);
        if (!osnet_engine) throw runtime_error("Failed to load OSNet engine");

        IExecutionContext* yolo_context = yolo_engine->createExecutionContext();
        if (!yolo_context) throw runtime_error("Failed to create YOLO context");

        IExecutionContext* osnet_context = osnet_engine->createExecutionContext();
        if (!osnet_context) throw runtime_error("Failed to create OSNet context");

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Get YOLO input/output names (assuming one input and one output)
        string yolo_input_name, yolo_output_name;
        for (int i = 0; i < yolo_engine->getNbIOTensors(); ++i) {
            const char* name = yolo_engine->getIOTensorName(i);
            if (yolo_engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                yolo_input_name = name;
            } else if (yolo_engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT) {
                yolo_output_name = name;
            }
        }
        if (yolo_input_name.empty() || yolo_output_name.empty()) throw runtime_error("Invalid YOLO engine IO");

        // Get OSNet input/output names
        string osnet_input_name, osnet_output_name;
        for (int i = 0; i < osnet_engine->getNbIOTensors(); ++i) {
            const char* name = osnet_engine->getIOTensorName(i);
            if (osnet_engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                osnet_input_name = name;
            } else if (osnet_engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT) {
                osnet_output_name = name;
            }
        }
        if (osnet_input_name.empty() || osnet_output_name.empty()) throw runtime_error("Invalid OSNet engine IO");

        // Assume FP32 I/O for simplicity; if FP16, you'll need to use __half and convert (add checks if needed)
        if (yolo_engine->getTensorDataType(yolo_input_name.c_str()) != DataType::kFLOAT ||
            yolo_engine->getTensorDataType(yolo_output_name.c_str()) != DataType::kFLOAT) {
            cerr << "Warning: Assuming FP32 I/O; adjust for FP16 if engine requires it" << endl;
        }
        if (osnet_engine->getTensorDataType(osnet_input_name.c_str()) != DataType::kFLOAT ||
            osnet_engine->getTensorDataType(osnet_output_name.c_str()) != DataType::kFLOAT) {
            cerr << "Warning: Assuming FP32 I/O; adjust for FP16 if engine requires it" << endl;
        }

        // Allocate YOLO buffers
        Dims yolo_input_dims = yolo_context->getTensorShape(yolo_input_name.c_str());
        int yolo_input_size = 1;
        for (int d = 0; d < yolo_input_dims.nbDims; ++d) yolo_input_size *= yolo_input_dims.d[d];
        size_t yolo_input_bytes = yolo_input_size * sizeof(float);
        void* yolo_input_gpu = nullptr;
        cudaMalloc(&yolo_input_gpu, yolo_input_bytes);

        Dims yolo_output_dims = yolo_context->getTensorShape(yolo_output_name.c_str());
        int yolo_output_size = 1;
        for (int d = 0; d < yolo_output_dims.nbDims; ++d) yolo_output_size *= yolo_output_dims.d[d];
        size_t yolo_output_bytes = yolo_output_size * sizeof(float);
        void* yolo_output_gpu = nullptr;
        cudaMalloc(&yolo_output_gpu, yolo_output_bytes);

        // Allocate OSNet buffers
        Dims osnet_input_dims = osnet_context->getTensorShape(osnet_input_name.c_str());
        int osnet_input_size = 1;
        for (int d = 0; d < osnet_input_dims.nbDims; ++d) osnet_input_size *= osnet_input_dims.d[d];
        size_t osnet_input_bytes = osnet_input_size * sizeof(float);
        void* osnet_input_gpu = nullptr;
        cudaMalloc(&osnet_input_gpu, osnet_input_bytes);

        Dims osnet_output_dims = osnet_context->getTensorShape(osnet_output_name.c_str());
        int osnet_output_size = 1;
        for (int d = 0; d < osnet_output_dims.nbDims; ++d) osnet_output_size *= osnet_output_dims.d[d];
        size_t osnet_output_bytes = osnet_output_size * sizeof(float);
        void* osnet_output_gpu = nullptr;
        cudaMalloc(&osnet_output_gpu, osnet_output_bytes);

        Tracker tracker(TRACK_MAX_AGE, EMB_THRESH);

        // Initialize video capture based on USE_WEBCAM flag
        cv::VideoCapture cap;
        if (USE_WEBCAM) {
            cap.open(0); // Webcam
        } else {
            cap.open(VIDEO_PATH); // Video file
        }
        if (!cap.isOpened()) {
            cerr << "Error: cannot open " << (USE_WEBCAM ? "webcam" : "video: " + VIDEO_PATH) << endl;
            return -1;
        }

        double prev_time = (double)cv::getTickCount();
        cv::Mat frame;
        while (cap.read(frame)) {
            // Preprocess
            vector<float> y_input;
            float ratio; int dw, dh;
            preprocess_yolo(frame, y_input, ratio, dw, dh);

            // Copy input to GPU and run YOLO
            cudaMemcpyAsync(yolo_input_gpu, y_input.data(), y_input.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
            yolo_context->setTensorAddress(yolo_input_name.c_str(), yolo_input_gpu);
            yolo_context->setTensorAddress(yolo_output_name.c_str(), yolo_output_gpu);
            yolo_context->enqueueV3(stream);
            cudaStreamSynchronize(stream);

            // Copy output back
            vector<float> raw(yolo_output_size);
            cudaMemcpyAsync(raw.data(), yolo_output_gpu, yolo_output_bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // Get shape
            vector<int64_t> shape(yolo_output_dims.nbDims);
            for (int i = 0; i < yolo_output_dims.nbDims; ++i) shape[i] = yolo_output_dims.d[i];

            // Postprocess
            vector<Box> boxes; vector<float> scores; vector<int> classids;
            postprocess_yolo(raw, shape, frame.size(), ratio, dw, dh, boxes, scores, classids);

            // Embeddings for each detection
            vector<vector<float>> embeddings;
            for (auto &b: boxes) {
                int x1 = (int)round(b.x1), y1 = (int)round(b.y1), x2 = (int)round(b.x2), y2 = (int)round(b.y2);
                x1 = max(0, min(x1, frame.cols - 1));
                x2 = max(0, min(x2, frame.cols - 1));
                y1 = max(0, min(y1, frame.rows - 1));
                y2 = max(0, min(y2, frame.rows - 1));
                if (x2 <= x1 || y2 <= y1) continue;
                cv::Mat crop = frame(cv::Rect(x1,y1,x2-x1,y2-y1)).clone();
                vector<float> emb = get_embedding(crop, osnet_context, stream, osnet_input_name, osnet_output_name, osnet_input_gpu, osnet_output_gpu, osnet_output_size);
                if (!emb.empty()) embeddings.push_back(emb);
                else embeddings.push_back(vector<float>());
            }

            // Trim boxes if embeddings are missing
            if (embeddings.size() != boxes.size()) {
                vector<Box> boxes2; vector<vector<float>> emb2;
                for (size_t i=0;i<boxes.size() && i<embeddings.size();++i) {
                    if (!embeddings[i].empty()) { boxes2.push_back(boxes[i]); emb2.push_back(embeddings[i]); }
                }
                boxes.swap(boxes2); embeddings.swap(emb2);
            }

            // Update tracker
            auto tracks = tracker.update(boxes, embeddings);

            // Draw tracks
            for (auto &t: tracks) {
                int x1 = (int)round(t.bbox.x1), y1 = (int)round(t.bbox.y1), x2 = (int)round(t.bbox.x2), y2 = (int)round(t.bbox.y2);
                cv::rectangle(frame, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(0,255,0), 2);
                string label = "ID:" + to_string(t.track_id);
                cv::putText(frame, label, cv::Point(x1, max(0,y1-5)), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
            }

            // FPS
            double curr_time = (double)cv::getTickCount();
            double fps = cv::getTickFrequency() / (curr_time - prev_time + 1e-9);
            prev_time = curr_time;
            char fpsbuf[64]; snprintf(fpsbuf, sizeof(fpsbuf), "FPS: %.1f", fps);
            cv::putText(frame, fpsbuf, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,255), 2);

            cv::imshow("UAV Tracker", frame);
            if (cv::waitKey(1) == 27) break;
        }

        // Cleanup
        cap.release();
        cv::destroyAllWindows();
        cudaFree(yolo_input_gpu);
        cudaFree(yolo_output_gpu);
        cudaFree(osnet_input_gpu);
        cudaFree(osnet_output_gpu);
        cudaStreamDestroy(stream);
        delete yolo_context;
        delete osnet_context;
        delete yolo_engine;
        delete osnet_engine;
        delete runtime;

    } catch (const exception &e) {
        cerr << "Exception: " << e.what() << endl;
        return -1;
    }

    return 0;
}