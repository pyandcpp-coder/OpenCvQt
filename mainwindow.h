// #ifndef MAINWINDOW_H
// #define MAINWINDOW_H

// #include <QMainWindow>
// #include <QTimer>
// #include <QLabel>
// #include <vector>
// #include <opencv2/opencv.hpp>
// #include <opencv2/dnn.hpp>
// #include <opencv2/dnn/all_layers.hpp>

// // Structure to store detection information
// struct Detection {
//     int classId;
//     float confidence;
//     cv::Rect box;
//     cv::Point center;
//     std::string className;
// };

// class MainWindow : public QMainWindow
// {
//     Q_OBJECT

// public:
//     MainWindow(QWidget *parent = nullptr);
//     ~MainWindow();

// private slots:
//     void processFrame();

// private:
//     // UI Components
//     QLabel *imageLabel;
//     QTimer *timer;

//     // OpenCV Components
//     cv::VideoCapture cap;
//     cv::dnn::Net net;
//     std::vector<std::string> classes;

//     // Parameters for detection
//     float confThreshold;
//     float nmsThreshold;
//     int inpWidth;
//     int inpHeight;

//     // Calibration variables
//     double pixelsPerCm;
//     bool isCalibrated;
//     double referenceWidthCm;
//     bool calibrationMode;

//     // Functions
//     std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);
//     std::vector<std::string> loadClassNames(const std::string& filename);
//     void drawPred(const Detection& detection, cv::Mat& frame);
//     double calculatePixelDistance(const cv::Point& p1, const cv::Point& p2);
//     double pixelToRealDistance(double pixelDistance, double pixelsPerCm);
//     double calibratePixelsPerCm(int referenceWidthPixels, double referenceWidthCm);
//     void setupNetwork();
//     void setupUI();
// };

// #endif // MAINWINDOW_H
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <QSpinBox>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Struct to store orientation information
struct OrientationInfo {
    float angle;             // Angle in degrees
    std::string direction;   // Direction description (e.g., "Horizontal", "Vertical")
    double confidence;       // Confidence in orientation detection
};

// Struct to store detection information
struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
    cv::Point center;
    std::string className;
    double width;            // Width in cm
    double height;           // Height in cm
    OrientationInfo orientation;
    double horizontalAngle;  // Angle with respect to horizontal
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void processFrame();
    void updateCalibration(int value);

private:
    // UI components
    QLabel *imageLabel;
    QSpinBox *calibrationInput;
    QTimer *timer;

    // OpenCV components
    cv::VideoCapture cap;
    cv::dnn::Net net;
    cv::Mat currentFrame;
    std::vector<std::string> classes;

    // Detection parameters
    float confThreshold;
    float nmsThreshold;
    int inpWidth;
    int inpHeight;
    double pixelsPerCm;
    bool isCalibrated;

    // Setup functions
    void setupUI();
    void setupNetwork();

    // Helper functions
    void drawPred(const Detection& detection, cv::Mat& frame);
    std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);
    std::vector<std::string> loadClassNames(const std::string& filename);
    double calculatePixelDistance(const cv::Point& p1, const cv::Point& p2);
    cv::Rect ensureRectInsideImage(const cv::Rect& rect, const cv::Mat& image);
    OrientationInfo determineOrientation(const cv::Mat& objectROI);
    double calculateHorizontalAngle(const cv::Rect& box);
};

#endif // MAINWINDOW_H
