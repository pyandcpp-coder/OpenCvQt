QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17
# QT +=widget

INCLUDEPATH += /usr/local/include/opencv4  # Keep this part
INCLUDEPATH += /usr/local/include/opencv4/opencv2 # ADD this line - more specific path

LIBS += -L/usr/local/lib \
        -lopencv_core \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lopencv_highgui \
        -lopencv_dnn \
        -lopencv_videoio  # Added opencv_videoio here



SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui



# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

