
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <QSpinBox>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct OrientationInfo {
    float angle;             
    std::string direction;   
    double confidence;       
};

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
    cv::Point center;
    std::string className;
    double width;            
    double height;          
    OrientationInfo orientation;
    double horizontalAngle;  
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
