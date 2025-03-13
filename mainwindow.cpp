
#include "mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QMessageBox>
#include <QFileDialog>
#include <iostream>
#include <fstream>
#include <cmath>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , confThreshold(0.5)
    , nmsThreshold(0.4)
    , inpWidth(416)
    , inpHeight(416)
    , pixelsPerCm(35)  // Default calibration
    , isCalibrated(true)
{
    setupUI();
    setupNetwork();

    // Start the timer to process frames
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::processFrame);
    timer->start(30); // Update at approximately 30 fps
}

MainWindow::~MainWindow()
{
    if (timer->isActive())
        timer->stop();

    if (cap.isOpened())
        cap.release();
}

void MainWindow::setupUI()
{
    // Set window properties
    setWindowTitle("Object Dimension and Orientation Measurement");
    resize(800, 600);

    // Create central widget and main layout
    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    setCentralWidget(centralWidget);

    // Create and add image labels
    imageLabel = new QLabel(this);
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    mainLayout->addWidget(imageLabel);

    // Create button layout
    QHBoxLayout *buttonLayout = new QHBoxLayout();

    // Add calibration input
    QLabel *calibrationLabel = new QLabel("Pixels per cm:", this);
    buttonLayout->addWidget(calibrationLabel);

    calibrationInput = new QSpinBox(this);
    calibrationInput->setRange(1, 200);
    calibrationInput->setValue(pixelsPerCm);
    connect(calibrationInput, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::updateCalibration);
    buttonLayout->addWidget(calibrationInput);

    mainLayout->addLayout(buttonLayout);
}

void MainWindow::updateCalibration(int value)
{
    pixelsPerCm = value;
    std::cout << "Calibration updated: " << pixelsPerCm << " pixels/cm" << std::endl;
}

void MainWindow::setupNetwork()
{
    // Use fixed paths to the model files
    std::string configPath = "/Users/yrevash/Downloads/yolov3.cfg";
    std::string weightsPath = "/Users/yrevash/Downloads/yolov3.weights";
    std::string classesPath = "/Users/yrevash/Downloads/coco.names";

    // Load the network
    try {
        net = cv::dnn::readNetFromDarknet(configPath, weightsPath);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to load the neural network: " << e.what() << std::endl;
        return;
    }

    // Load the class names
    classes = loadClassNames(classesPath);
    if (classes.empty()) {
        std::cerr << "Failed to load class names from: " << classesPath << std::endl;
        return;
    }

    // Open camera
    cap.open(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera." << std::endl;
        return;
    }
}

cv::Rect MainWindow::ensureRectInsideImage(const cv::Rect& rect, const cv::Mat& image) {
    cv::Rect result = rect;
    // Make sure x and width are within bounds
    result.x = std::max(0, result.x);
    result.width = std::min(image.cols - result.x, result.width);

    // Make sure y and height are within bounds
    result.y = std::max(0, result.y);
    result.height = std::min(image.rows - result.y, result.height);

    return result;
}

void MainWindow::processFrame()
{
    if (!cap.read(currentFrame)) {
        std::cerr << "Failed to grab frame from camera" << std::endl;
        return;
    }

    // Create a copy for drawing
    cv::Mat displayFrame = currentFrame.clone();

    // Create a 4D blob from the frame
    cv::Mat blob;
    cv::dnn::blobFromImage(currentFrame, blob, 1/255.0, cv::Size(inpWidth, inpHeight),
                           cv::Scalar(0, 0, 0), true, false);

    // Set the input to the network
    net.setInput(blob);

    // Forward pass to get output
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Process detection results
    std::vector<Detection> detections;

    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;

            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            // Check if the detection has sufficient confidence
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * currentFrame.cols);
                int centerY = (int)(data[1] * currentFrame.rows);
                int width = (int)(data[2] * currentFrame.cols);
                int height = (int)(data[3] * currentFrame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                Detection detection;
                detection.classId = classIdPoint.x;
                detection.confidence = (float)confidence;
                detection.box = cv::Rect(left, top, width, height);

                // Ensure the bounding box is within image boundaries
                detection.box = ensureRectInsideImage(detection.box, currentFrame);

                // Recalculate center after adjusting the bounding box
                detection.center = cv::Point(detection.box.x + detection.box.width / 2,
                                             detection.box.y + detection.box.height / 2);

                if (detection.classId < classes.size())
                    detection.className = classes[classIdPoint.x];
                else
                    detection.className = "Unknown";

                // Calculate real-world dimensions
                detection.width = detection.box.width / pixelsPerCm;
                detection.height = detection.box.height / pixelsPerCm;

                // Calculate orientation based on aspect ratio and shape analysis
                // Make sure we're passing a valid ROI
                if (detection.box.width > 0 && detection.box.height > 0) {
                    detection.orientation = determineOrientation(currentFrame(detection.box));

                    // Calculate angle wrt horizontal
                    detection.horizontalAngle = calculateHorizontalAngle(detection.box);
                } else {
                    // Default orientation if box dimensions are invalid
                    detection.orientation.angle = 0;
                    detection.orientation.direction = "Unknown";
                    detection.orientation.confidence = 0;
                    detection.horizontalAngle = 0;
                }

                detections.push_back(detection);
            }
        }
    }

    // Apply non-maximum suppression to eliminate redundant overlapping boxes
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    for (const auto& detection : detections) {
        boxes.push_back(detection.box);
        confidences.push_back(detection.confidence);
    }

    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    // Draw detections with dimension and orientation information
    for (size_t i = 0; i < indices.size(); ++i) {
        drawPred(detections[indices[i]], displayFrame);
    }

    // Display calibration information
    std::string calibrationInfo = "Calibration: " + std::to_string(pixelsPerCm) + " pixels/cm";
    cv::putText(displayFrame, calibrationInfo, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    // Display the objects count
    std::string objectCountInfo = "Objects detected: " + std::to_string(indices.size());
    cv::putText(displayFrame, objectCountInfo, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    // Convert the image to Qt format and display it
    QImage qimg(displayFrame.data, displayFrame.cols, displayFrame.rows, displayFrame.step, QImage::Format_BGR888);
    QPixmap pixmap = QPixmap::fromImage(qimg);
    imageLabel->setPixmap(pixmap.scaled(imageLabel->size(), Qt::KeepAspectRatio));
}

OrientationInfo MainWindow::determineOrientation(const cv::Mat& objectROI)
{
    OrientationInfo info;
    info.angle = 0;
    info.direction = "Unknown";
    info.confidence = 0;

    // Check if ROI is valid
    if (objectROI.empty() || objectROI.rows <= 0 || objectROI.cols <= 0) {
        return info;
    }

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(objectROI, gray, cv::COLOR_BGR2GRAY);

    // Threshold to get binary image
    cv::Mat binary;
    cv::threshold(gray, binary, 128, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // If no contours found, return default
    if (contours.empty()) {
        return info;
    }

    // Find the largest contour
    size_t largestContourIdx = 0;
    double largestArea = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > largestArea) {
            largestArea = area;
            largestContourIdx = i;
        }
    }

    // Check if the largest contour has enough points
    if (contours[largestContourIdx].size() < 5) {
        return info;
    }

    // Find the minimum area rectangle
    cv::RotatedRect minRect = cv::minAreaRect(contours[largestContourIdx]);

    // Get the angle
    float angle = minRect.angle;

    // The angle is between -90 and 0 degrees, we add 90 to make it 0-90
    if (angle < -45)
        angle += 90;

    // Determine the "pointiness" of the object to guess direction
    // This is a simplified approach - in a real application you'd need more sophisticated methods
    cv::Point2f vertices[4];
    minRect.points(vertices);

    // Calculate the length of each side
    float side1 = cv::norm(vertices[0] - vertices[1]);
    float side2 = cv::norm(vertices[1] - vertices[2]);

    // Use aspect ratio to determine front/back
    float aspectRatio = std::max(side1, side2) / std::min(side1, side2);

    // Simple heuristic: if aspect ratio is high enough, it's directional
    if (aspectRatio > 2.0) {
        info.confidence = std::min(1.0, (aspectRatio - 2.0) / 3.0);
    } else {
        info.confidence = 0.0;
    }

    // For direction, we'll use the angle of the rectangle
    // This is a simplified approach - for real objects you'd need more analysis
    if (info.confidence > 0.3) {
        // Angle would determine left/right or up/down
        if (abs(angle) < 45) {
            info.direction = "Horizontal";
        } else {
            info.direction = "Vertical";
        }
    } else {
        info.direction = "Unknown";
    }

    info.angle = angle;
    return info;
}

double MainWindow::calculateHorizontalAngle(const cv::Rect& box)
{
    // Calculate height to width ratio
    double h = box.height;
    double w = box.width;

    // A simple way to determine angle with horizontal
    // This is a heuristic - for real applications you might need more sophisticated analysis
    double angle = std::atan2(h, w) * 180 / M_PI;

    return angle;
}

void MainWindow::drawPred(const Detection& detection, cv::Mat& frame) {
    int left = detection.box.x;
    int top = detection.box.y;
    int right = detection.box.x + detection.box.width;
    int bottom = detection.box.y + detection.box.height;

    // Draw a rectangle around the detected object
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);

    // Mark the center point
    cv::circle(frame, detection.center, 5, cv::Scalar(0, 0, 255), -1);

    // Create the label with class name, confidence, and dimensions
    std::string label = detection.className + ": " +
                        cv::format("%.2f", detection.confidence) +
                        cv::format(" (%.1fcm x %.1fcm)", detection.width, detection.height);

    // Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
    top = std::max(top, labelSize.height);
    cv::putText(frame, label, cv::Point(left, top - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    // Draw orientation arrow
    if (detection.orientation.confidence > 0.3) {
        // Calculate arrow endpoint
        double angle = detection.orientation.angle * M_PI / 180; // Convert to radians
        int arrowLength = std::min(detection.box.width, detection.box.height) / 2;
        cv::Point arrowEnd = detection.center + cv::Point(
                                 static_cast<int>(arrowLength * cos(angle)),
                                 static_cast<int>(arrowLength * sin(angle))
                                 );

        // Draw the arrow
        cv::arrowedLine(frame, detection.center, arrowEnd, cv::Scalar(0, 255, 255), 2, cv::LINE_AA, 0, 0.3);

        // Display orientation info
        std::string orientationText = cv::format("Dir: %s", detection.orientation.direction.c_str());
        cv::putText(frame, orientationText, cv::Point(left, top + 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }

    // Display horizontal angle
    std::string angleText = cv::format("Angle: %.1f°", detection.horizontalAngle);
    cv::putText(frame, angleText, cv::Point(left, top + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

std::vector<std::string> MainWindow::getOutputsNames(const cv::dnn::Net& net) {
    static std::vector<std::string> names;
    if (names.empty()) {
        // Get the indices of the output layers
        std::vector<int> outLayers = net.getUnconnectedOutLayers();

        // Get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();

        // Get the names of the output layers
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

std::vector<std::string> MainWindow::loadClassNames(const std::string& filename) {
    std::vector<std::string> classNames;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening class names file: " << filename << std::endl;
        return classNames;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty())
            classNames.push_back(line);
    }
    return classNames;
}

double MainWindow::calculatePixelDistance(const cv::Point& p1, const cv::Point& p2) {
    return std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
}

