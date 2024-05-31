#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <CLI/CLI.hpp>

#include <chrono>
#include <format>
#include <iostream>
#include <optional>
#include <variant>

using namespace cv;

struct AppState {
    double zoomFactor = 1.0;
    Mat cflow, originalCflow;
    std::vector<Mat> images;
    std::vector<Mat> flowMaps;
    std::vector<Mat> flowVisualisations;
    std::string windowName = "Optical Flow";
    size_t currentFrame = 0;
    size_t numFrames = 0;
    double threshold = 0;
};

template<typename T>
concept DenseOpticalFlowAlgorithm = requires(
    const T t,
    const cv::cuda::GpuMat& frame1,
    const cv::cuda::GpuMat& frame2,
    const cv::cuda::GpuMat& flow
) {
    { t.calculateFlow(frame1, frame2, flow) } -> std::same_as<void>;
};

template<typename T>
concept SparseOpticalFlowAlgorithm = requires(
    const T t,
    const cv::cuda::GpuMat& frame1,
    const cv::cuda::GpuMat& frame2
) {
    { t.trackPoints(frame1, frame2)} -> std::same_as<void>;
};

class Farneback {
    Ptr<cv::cuda::FarnebackOpticalFlow> farnebackFlow;
    int numLevels = 3;
    bool fastPyramids = true;
public:
    Farneback() : farnebackFlow(cv::cuda::FarnebackOpticalFlow::create()) {
        farnebackFlow->setFastPyramids(fastPyramids);
        farnebackFlow->setNumLevels(numLevels);
    }
    void calculateFlow(const cv::cuda::GpuMat& frame1, const cv::cuda::GpuMat& frame2, const cv::cuda::GpuMat& flow) const {
        farnebackFlow->calc(frame1, frame2, flow);
    }
};

class LKSparse {
    Ptr<cv::cuda::ORB> orb;
    Ptr<cuda::SparsePyrLKOpticalFlow> lk;
    cv::cuda::GpuMat statusGPU;
    cv::cuda::GpuMat descriptors;
    std::vector<KeyPoint> keyPoints;
    std::vector<KeyPoint> nextPoints;
public:
    LKSparse(): orb(cv::cuda::ORB::create()), lk(cv::cuda::SparsePyrLKOpticalFlow::create()) {}
    void run(const cv::cuda::GpuMat& frame1, const cv::cuda::GpuMat& frame2) {
        orb->detectAndCompute(frame1, cv::cuda::GpuMat(), keyPoints, descriptors);
        lk->calc(frame1, frame2, keyPoints, nextPoints, statusGPU);
    }
};

class LK {
    Ptr<cv::cuda::DensePyrLKOpticalFlow> LKflow;
public:
    LK() : LKflow(cv::cuda::DensePyrLKOpticalFlow::create()) {}
    void calculateFlow(const cv::cuda::GpuMat& frame1, const cv::cuda::GpuMat& frame2, const cv::cuda::GpuMat& flow) const {
        LKflow->calc(frame1, frame2, flow);
    }
};

class NvidiaFlow {
    Ptr<cv::cuda::NvidiaOpticalFlow_2_0> nvidiaFlow;
public:
    NvidiaFlow(Size size) {}
    void calculateFlow(const cv::cuda::GpuMat& frame1, const cv::cuda::GpuMat& frame2, const cv::cuda::GpuMat& flow) const {
        nvidiaFlow->calc(frame1, frame2, flow);
    }
};

using OpticalFlowVariant = std::variant<
    Farneback,
    LK,
    LKSparse,
    NvidiaFlow
>;

void calculateOpticalFlow(
    const OpticalFlowVariant& algorithm,
    const cv::cuda::GpuMat& frame1,
    const cv::cuda::GpuMat& frame2,
    cv::cuda::GpuMat& flow
)  {
    std::visit([&](const auto& algo) {
        if constexpr(DenseOpticalFlowAlgorithm<decltype(algo)>) {
            algo.calculateFlow(frame1, frame2, flow);
        } else if constexpr(SparseOpticalFlowAlgorithm<decltype(algo)>) {
            algo.trackPoints(frame1, frame2);
        }
    },
    algorithm);
}

enum class FlowAlgo : int {
    Farneback,
    LK,
    Nvidia
};

std::ostream& operator<<(std::ostream& os, FlowAlgo algo) {
    switch (algo) {
        case FlowAlgo::Farneback:   os << "Farneback";     break;
        case FlowAlgo::LK:          os << "Lucas-Kanade";  break;
        case FlowAlgo::Nvidia:      os << "Nvidia";        break;
        default:                    os << "Unknown";       break;
    }
    return os;
}

std::unique_ptr<OpticalFlowVariant> createOpticalFlowAlgorithm(FlowAlgo algo, Size size) {
    switch (algo) {
        case FlowAlgo::Farneback:
            return std::make_unique<OpticalFlowVariant>(Farneback{});
        case FlowAlgo::LK:
            return std::make_unique<OpticalFlowVariant>(LK{});
        case FlowAlgo::Nvidia:
            return std::make_unique<OpticalFlowVariant>(NvidiaFlow(size));
        default:
            throw std::invalid_argument("Unknown FlowAlgo");
    }
}

// Function to draw optical flow vectors
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color, std::optional<double> threshold = std::nullopt) {
    double maxRad = -1;

    // First pass to find the maximum magnitude (radial distance)
    for (int y = 0; y < cflowmap.rows; y += step) {
        for (int x = 0; x < cflowmap.cols; x += step) {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            double rad = sqrt(fxy.x * fxy.x + fxy.y * fxy.y);
            if (rad > maxRad) maxRad = rad;
        }
    }

    // Second pass to draw the vectors, scaled by the maximum magnitude
    for (int y = 0; y < cflowmap.rows; y += step) {
        for (int x = 0; x < cflowmap.cols; x += step) {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            double rad = sqrt(fxy.x * fxy.x + fxy.y * fxy.y);
            if (!threshold.has_value() || rad > threshold.value()) { // Apply threshold to filter out small vectors
                Point2f scaledFxy = fxy * (10.0 / maxRad); // scale the vectors
                Point center(x + step / 2, y + step / 2); // Center of the cell
                line(cflowmap, center, Point(cvRound(center.x + scaledFxy.x), cvRound(center.y + scaledFxy.y)), color);
                circle(cflowmap, center, 2, color, -1);
            }
        }
    }
}

void drawKeyPoints(const std::vector<KeyPoint> points) {

}

void redraw(AppState& appState) {
    std::cout << std::format("Redrawing frame {}", appState.currentFrame) << std::endl;
    appState.originalCflow = appState.flowVisualisations[appState.currentFrame].clone();
    resize(appState.originalCflow, appState.cflow, Size(), appState.zoomFactor, appState.zoomFactor, INTER_LINEAR);
    imshow(appState.windowName, appState.cflow);
}

void onMouse(int event, int x, int y, int flags, void* userData) {
    AppState* appState = static_cast<AppState*>(userData);
    if (event == EVENT_MOUSEWHEEL) {
        if (getMouseWheelDelta(flags) > 0) {
            appState->zoomFactor *= 1.1;
        } else {
            appState->zoomFactor /= 1.1;
        }

        // Resize the original cflow image based on the zoom factor
        resize(appState->originalCflow, appState->cflow, Size(), appState->zoomFactor, appState->zoomFactor, INTER_LINEAR);
        
        // Display the resized image
        imshow(appState->windowName, appState->cflow);
    }
}

enum class KeyAction : int {
    NextFrame = 'd',
    PrevFrame = 'a',
    IncreaseThreshold = '+',
    DecreaseThreshold = '-'
};

void onKey(AppState& appState, KeyAction key) {
    switch (key) {
        case KeyAction::NextFrame:
            if (appState.currentFrame > 0) {
                appState.currentFrame--;
                redraw(appState);
            }
            break;
        case KeyAction::PrevFrame:
            if (appState.currentFrame < appState.numFrames - 2) {
                appState.currentFrame++;
                redraw(appState);
            }
            break;
        case KeyAction::IncreaseThreshold:
            std::cout << std::format("Increasing threshold to {}", 0) << std::endl;
            break;
        case KeyAction::DecreaseThreshold:
            std::cout << std::format("Decreasing threshold to {}", 0) << std::endl;
            break;
        default:
            break;
    }
}

std::string DEMO_IMAGE_DIR = "C:\\dev\\data\\Test Images\\";
std::string CLOCK_VIDEO_8BIT_PATH = DEMO_IMAGE_DIR + "clock_video_8bit.tiff";

int main(int argc, char **argv) {
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) 
        std::cout << "CUDA is enabled and devices are available." << std::endl;
    else {
        std::cout << "CUDA is not enabled or no CUDA devices are available." << std::endl;
        return 1;
    }

    AppState appState;
    
    CLI::App app;
    FlowAlgo flowAlgo;
    
    std::map<std::string, FlowAlgo> flowAlgoMap {
        {"Farneback",   FlowAlgo::Farneback},
        {"LK",          FlowAlgo::LK},
        {"Nvidia",      FlowAlgo::Nvidia}
    };

    app.add_option("-a,--algorithm", flowAlgo, "Optical flow algorithm to use")
        ->required()
        ->transform(CLI::CheckedTransformer(flowAlgoMap, CLI::ignore_case));

    app.add_option("-n,--numframes", appState.numFrames, "Number of frames to process for optical flow visualization")
        ->required()
        ->check(CLI::PositiveNumber);

    CLI11_PARSE(app, argc, argv);

    std::cout << "Using algorithm: " << flowAlgo << std::endl;

    // Step 1: Read the multipage TIFF file
    cv::imreadmulti(CLOCK_VIDEO_8BIT_PATH, appState.images, cv::IMREAD_GRAYSCALE);

    // Upload frames to GPU
    cv::cuda::GpuMat gpu_frame1, gpu_frame2;
    Size size = appState.images[0].size();
    cv::cuda::GpuMat gpuFlow(size, CV_32FC2);

    std::unique_ptr<OpticalFlowVariant> optFlow = createOpticalFlowAlgorithm(flowAlgo, gpu_frame1.size());

    for (size_t i = 1; i < appState.numFrames; ++i) {
        gpu_frame1.upload(appState.images[i - 1]);
        gpu_frame2.upload(appState.images[i]);

        auto start = std::chrono::high_resolution_clock::now();
        calculateOpticalFlow(*optFlow, gpu_frame1, gpu_frame2, gpuFlow);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        std::cout << "Time for optical flow calculation between frame " << i - 1 << " and " << i << ": " << duration.count() << " ms" << std::endl;

        Mat flow(gpuFlow);
        appState.flowMaps.push_back(flow);
        Mat flowVis;
        cvtColor(appState.images[i - 1], flowVis, COLOR_GRAY2BGR);
        double noiseThreshold = 7.5; // Threshold to filter out noise
        drawOptFlowMap(flow, flowVis, 16, Scalar(0, 255, 0), noiseThreshold);
        appState.flowVisualisations.push_back(flowVis);
    }

    // Set the initial display image
    appState.originalCflow = appState.flowVisualisations[appState.currentFrame].clone();
    appState.cflow = appState.originalCflow.clone();

    // Create a window and set the mouse callback
    namedWindow(appState.windowName, WINDOW_AUTOSIZE);
    setMouseCallback(appState.windowName, onMouse, &appState);
    
    // Show the result
    imshow(appState.windowName, appState.cflow);

    while (true) {
        int key = waitKey(0);
        if (key == 27) {
            break;
        }
        onKey(appState, static_cast<KeyAction>(key));
    }

    return 0;
}
