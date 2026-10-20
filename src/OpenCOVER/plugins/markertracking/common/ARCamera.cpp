#include "ARCamera.h"

#include <OpenConfig/value.h>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/calib3d.hpp>
#else
#include <opencv2/calib3d/calib3d.hpp>
#endif

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <utility>

using opencover::cover;

namespace
{
#ifdef __linux__
std::string runCommand(const std::string &command)
{
    std::array<char, 512> buffer { };
    std::string output;
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe)
        return output;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe))
        output += buffer.data();
    pclose(pipe);
    return output;
}

using namespace opencover;

std::string parseCardType(const std::string &v4l2Info)
{
    std::istringstream stream(v4l2Info);
    std::string line;
    while (std::getline(stream, line))
    {
        auto pos = line.find("Card type");
        if (pos == std::string::npos)
            continue;
        auto colon = line.find(':', pos);
        if (colon == std::string::npos)
            continue;
        std::string name = line.substr(colon + 1);
        while (!name.empty() && std::isspace(static_cast<unsigned char>(name.front())))
            name.erase(name.begin());
        while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back())))
            name.pop_back();
        return name;
    }
    return "unknown";
}
#endif

std::vector<int> cameraBackends()
{
#ifdef _WIN32
    return { cv::CAP_MSMF, cv::CAP_DSHOW, cv::CAP_ANY };
#elif defined(__APPLE__)
    return { cv::CAP_AVFOUNDATION, cv::CAP_ANY };
#else
    return { cv::CAP_V4L2, cv::CAP_ANY };
#endif
}
}

std::vector<ARCamera::Device> ARCamera::availableDevices()
{
    std::vector<Device> devices;
#ifdef __linux__
    std::error_code error;
    const std::filesystem::path deviceLinks("/dev/v4l/by-id");
    constexpr std::string_view captureNodeSuffix = "-video-index0";
    for (const auto &entry : std::filesystem::directory_iterator(deviceLinks, error))
    {
        const std::string filename = entry.path().filename().string();
        if (filename.size() <= captureNodeSuffix.size() || filename.compare(filename.size() - captureNodeSuffix.size(), captureNodeSuffix.size(), captureNodeSuffix) != 0)
            continue;

        const std::filesystem::path devicePath = std::filesystem::canonical(entry.path(), error);
        if (error)
            continue;
        const std::string deviceName = devicePath.filename().string();
        if (deviceName.rfind("video", 0) != 0)
            continue;

        const std::string number = deviceName.substr(5);
        if (number.empty() || !std::all_of(number.begin(), number.end(), [](unsigned char ch)
                { return std::isdigit(ch); }))
            continue;

        const int deviceId = std::stoi(number);
        const std::string info = runCommand("v4l2-ctl -d " + devicePath.string() + " --info 2>/dev/null");
        if (!info.empty())
            devices.push_back({ deviceId, std::to_string(deviceId) + " - " + parseCardType(info) });
    }
    std::sort(devices.begin(), devices.end(), [](const Device &left, const Device &right)
        { return left.id < right.id; });
#else
    constexpr int maxDeviceIndex = 16;
    const auto backends = cameraBackends();
    for (int deviceId = 0; deviceId < maxDeviceIndex; ++deviceId)
    {
        for (const int backend : backends)
        {
            cv::VideoCapture capture;
            try
            {
                if (!capture.open(deviceId, backend))
                    continue;

                cv::Mat frame;
                if (capture.read(frame) && !frame.empty())
                {
                    devices.push_back({ deviceId, "Camera " + std::to_string(deviceId) });
                    capture.release();
                    break;
                }
            }
            catch (const cv::Exception &)
            {
            }
            capture.release();
        }
    }
#endif
    return devices;
}

bool ARCamera::open(int deviceId)
{
    close();
    m_cameraMatrix.release();
    m_distortionCoefficients.release();
    m_deferredCalibration = false;
    m_calibrating = false;
    m_allCorners.clear();
    m_allIds.clear();
    m_allImages.clear();
    m_lastCalibrationCapture = 0.0;

    for (const int backend : cameraBackends())
        if (m_capture.open(deviceId, backend))
            break;

    if (!m_capture.isOpened())
        return false;

    const std::string fourcc = m_configFile->value<std::string>("camera", "fourcc", "")->value();
    if (fourcc.length() == 4)
    {
        m_capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]));
    }
    const double fps = m_configFile->value<double>("camera", "fps", 0.0)->value();
    if (fps > 0.f)
    {
        m_capture.set(cv::CAP_PROP_FPS, fps);
    }

    m_capture.set(cv::CAP_PROP_FRAME_WIDTH, m_width);
    m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, m_height);

    const int focus = static_cast<int>(m_configFile->value<int64_t>("camera", "focus", 35)->value());
    // Disable auto focus to avoid focus hunting during calibration
    const bool autofocusSet = m_capture.set(cv::CAP_PROP_AUTOFOCUS, 0);
    // Define manual focus
    const bool focusSet = m_capture.set(cv::CAP_PROP_FOCUS, focus);

    if (!autofocusSet || !focusSet)
    {
        std::cerr << "ARUCO WARNING: could not set camera focus to " << focus << std::endl;
    }

    cv::Mat frame;
    // Capture a first frame:
    try
    {
        m_capture >> frame;
    }
    catch (const cv::Exception &ex)
    {
        std::cerr << "OpenCV exception: " << ex.what() << std::endl;
    }
    if (frame.empty())
    {
        std::cerr << "ARUCO: camera device " << deviceId << " returned an empty frame (device not usable)." << std::endl;
        close();
        return false;
    }

    if (static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH)) != m_width || static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT)) != m_height)
    {
        std::cerr << "ARUCO WARNING: could not set capture frame size to  = " << m_width << "x" << m_height <<
        "\n Actual frame size is: " << frame.cols << "x" << frame.rows << std::endl;
    }

    m_deferredCalibration = !loadCalibration(frame);
    return true;
}

void ARCamera::close()
{
    if (m_capture.isOpened())
        m_capture.release();
}

bool ARCamera::isOpened() const
{
    return m_capture.isOpened();
}

bool ARCamera::read(cv::Mat &frame)
{
    if (!m_capture.isOpened())
        return false;
    m_capture >> frame;
    return !frame.empty();
}

bool ARCamera::setManualFocus(int focus)
{
    if (!m_capture.isOpened())
        return false;

    const bool autofocusSet = m_capture.set(cv::CAP_PROP_AUTOFOCUS, 0);
    const bool focusSet = m_capture.set(cv::CAP_PROP_FOCUS, focus);
    const double reportedFocus = m_capture.get(cv::CAP_PROP_FOCUS);
    if (!focusSet || (reportedFocus >= 0.0 && reportedFocus != focus))
    {
        std::cerr << " ARUCO WARNING: Camera did not confirm requested focus!" << std::endl;

        if (cover->debugLevel(2))
        {
            std::cerr << "ARUCO: focus set=" << focusSet
                      << ", autofocus disabled=" << autofocusSet
                      << ", requested focus=" << focus
                      << ", reported focus=" << reportedFocus << std::endl;
        }
    }
    return focusSet;
}

int ARCamera::width() const
{
    return m_width;
}

int ARCamera::height() const
{
    return m_height;
}

const cv::Mat &ARCamera::cameraMatrix() const
{
    return m_cameraMatrix;
}

const cv::Mat &ARCamera::distortionCoefficients() const
{
    return m_distortionCoefficients;
}

bool ARCamera::hasIntrinsics() const
{
    return !m_cameraMatrix.empty() && m_cameraMatrix.rows == 3 && m_cameraMatrix.cols == 3;
}

bool ARCamera::loadCalibration(const cv::Mat &frame)
{
    std::cerr << "ARUCO: loading calibration data from file " << m_calibrationFilename << std::endl;
    cv::FileStorage storage;
    try
    {
        storage.open(m_calibrationFilename, cv::FileStorage::READ);
    }
    catch (const cv::Exception &)
    {
    }

    if (storage.isOpened())
    {
        storage["camera_matrix"] >> m_cameraMatrix;
        storage["dist_coefs"] >> m_distortionCoefficients;
        if (hasIntrinsics() && !m_distortionCoefficients.empty())
            return true;

        std::cerr << "ARUCO: camera calibration file does not contain usable parameters" << std::endl;
        m_cameraMatrix.release();
        m_distortionCoefficients.release();
    }

    else
    {
        std::cerr << "ARUCO: failed to open camera calibration file " << m_calibrationFilename << std::endl;
    }
    std::cerr << "ARUCO: start calibration ... " << std::endl;
    const double focalLength = 250.0;
    const double cx = (frame.cols - 1) / 2.0;
    const double cy = (frame.rows - 1) / 2.0;
    m_cameraMatrix = (cv::Mat_<double>(3, 3) << focalLength, 0.0, cx,
        0.0, focalLength, cy,
        0.0, 0.0, 1.0);
    m_distortionCoefficients = cv::Mat::zeros(5, 1, CV_64F);
    return false;
}

bool ARCamera::consumeDeferredCalibrationRequest()
{
    const bool requested = m_deferredCalibration;
    m_deferredCalibration = false;
    return requested;
}

bool ARCamera::isCalibrating() const
{
    return m_calibrating;
}

void ARCamera::startCalibration()
{
    m_allCorners.clear();
    m_allIds.clear();
    m_allImages.clear();
    m_lastCalibrationCapture = 0.0;
    m_calibrating = true;
}

bool ARCamera::saveCalibration(const cv::Size &imageSize, float aspectRatio, int flags,
    const cv::Mat &cameraMatrix, const cv::Mat &distortionCoefficients,
    double reprojectionError) const
{
    const std::filesystem::path path(m_calibrationFilename);
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());

    cv::FileStorage storage(m_calibrationFilename, cv::FileStorage::WRITE);
    if (!storage.isOpened())
        return false;

    std::time_t now = std::time(nullptr);
    std::tm *localTime = std::localtime(&now);
    char buffer[1024];
    std::strftime(buffer, sizeof(buffer) - 1, "%c", localTime);
    storage << "calibration_time" << buffer;
    storage << "image_width" << imageSize.width;
    storage << "image_height" << imageSize.height;
    if (flags & cv::CALIB_FIX_ASPECT_RATIO)
        storage << "aspectRatio" << aspectRatio;
    storage << "flags" << flags;
    storage << "camera_matrix" << cameraMatrix;
    storage << "dist_coefs" << distortionCoefficients;
    storage << "avg_reprojection_error" << reprojectionError;
    return true;
}

void ARCamera::finishCalibration(const cv::Mat &cameraMatrix, const cv::Mat &distortionCoefficients,
    const cv::Size &imageSize, float aspectRatio, int flags,
    double reprojectionError)
{
    if (!saveCalibration(imageSize, aspectRatio, flags, cameraMatrix, distortionCoefficients, reprojectionError))
    {
        std::cerr << "ARUCO calib save FAILED: " << m_calibrationFilename << std::endl;
        return;
    }

    m_cameraMatrix = cameraMatrix;
    m_distortionCoefficients = distortionCoefficients;
    m_calibrating = false;
    m_allImages.clear();
    m_allIds.clear();
    m_allCorners.clear();
    std::cerr << "ARUCO calib saved: " << m_calibrationFilename << std::endl;
}

void ARCamera::calibrateFrame(const cv::Mat &frame,
    const std::vector<std::vector<cv::Point2f>> &markerCorners,
    const std::vector<int> &markerIds,
    const cv::Ptr<cv::aruco::CharucoDetector> &detector,
    const cv::Ptr<cv::aruco::CharucoBoard> &board)
{
    if (frame.empty() || !detector || !board || !m_calibrating)
        return;

    cv::Mat gray;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = frame;

    std::vector<cv::Point2f> charucoCorners;
    std::vector<int> charucoIds;
    // cv::Mat charucoCorners, charucoIds;
    auto refinedMarkerCorners = markerCorners;
    auto refinedMarkerIds = markerIds;
    detector->detectBoard(gray, charucoCorners, charucoIds, refinedMarkerCorners, refinedMarkerIds);

    if (cover->debugLevel(2))
    {
        std::cerr << "ARUCO calib: detected " << refinedMarkerIds.size() << " markers, "
                  << charucoIds.size() << " ChArUco corners" << std::endl;
    }

    const int numCorners = charucoIds.size();

    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    const bool captureIntervalElapsed = m_lastCalibrationCapture <= 0.0 || now - m_lastCalibrationCapture > 0.20;
    const bool sampleAccepted = numCorners >= 4 && captureIntervalElapsed;
    if (sampleAccepted)
    {
        m_lastCalibrationCapture = now;
        m_allCorners.push_back(refinedMarkerCorners);
        m_allIds.push_back(refinedMarkerIds);
        m_allImages.push_back(frame.clone());
        m_calibrationImageSize = frame.size();

        if (cover->debugLevel(2))
        {
            std::cerr << "ARUCO calib accepted: " << m_allIds.size() << " (charuco corners=" << numCorners << ")" << std::endl;
        }
    }

    if (cover->debugLevel(2))
    {
        cv::Mat preview = frame.clone(); // copy image for preview overlay, so that markers are not drawn to the image used for calibration
        if (numCorners > 0)
            cv::aruco::drawDetectedCornersCharuco(preview, charucoCorners, charucoIds);

        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Scalar laplacianMean, laplacianStdDev;
        cv::meanStdDev(laplacian, laplacianMean, laplacianStdDev);
        const double sharpness = laplacianStdDev[0] * laplacianStdDev[0];
        const double brightness = cv::mean(gray)[0];

        std::ostringstream status;
        if (sampleAccepted)
            status << "ACCEPTED sample " << m_allImages.size();
        else if (numCorners == 0)
            status << "NOT ACCEPTED: no ChArUco corners";
        else
            status << "WAITING: capture interval";

        std::ostringstream quality;
        quality << frame.cols << 'x' << frame.rows << " | ChArUco corners: " << numCorners
                << " | brightness: " << std::fixed << std::setprecision(1) << brightness
                << " | sharpness (Laplacian variance): " << std::setprecision(0) << sharpness;
        cv::putText(preview, status.str(), cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX,
            0.65, sampleAccepted ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 180, 255), 2);
        cv::putText(preview, quality.str(), cv::Point(12, 56), cv::FONT_HERSHEY_SIMPLEX,
            0.5, cv::Scalar(255, 255, 255), 1);
        try
        {
            cv::imshow("ARUCO calibration input", preview);
            cv::waitKey(1);
        }
        catch (const cv::Exception &exception)
        {
            std::cerr << "ARUCO calibration preview unavailable: " << exception.what() << std::endl;
        }
    }

    if (m_allIds.size() <= 15)
        return;

    cv::Mat cameraMatrix, distCoefficients;
    std::vector<cv::Mat> rotationVectors, translationVectors;
    double reprojectionError = -1.0;
    int calibrationFlags = 0;
    constexpr float aspectRatio = 1.0f;

#if (CV_VERSION_MAJOR >= 5) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
    const auto &boardCorners = board->getChessboardCorners();
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    for (size_t i = 0; i < m_allImages.size() && i < m_allCorners.size() && i < m_allIds.size(); ++i)
    {
        cv::Mat calibrationGray;
        if (m_allImages[i].channels() == 3)
            cv::cvtColor(m_allImages[i], calibrationGray, cv::COLOR_BGR2GRAY);
        else
            calibrationGray = m_allImages[i];

        std::vector<cv::Point2f> detectedCorners;
        std::vector<int> detectedIds;
        auto frameMarkerCorners = m_allCorners[i];
        auto frameMarkerIds = m_allIds[i];
        detector->detectBoard(calibrationGray, detectedCorners, detectedIds,
            frameMarkerCorners, frameMarkerIds);
        if (detectedCorners.empty() || detectedIds.empty())
            continue;

        std::vector<cv::Point2f> imagePointsForFrame;
        std::vector<cv::Point3f> objectPointsForFrame;
        for (size_t k = 0; k < detectedIds.size(); ++k)
        {
            const int id = detectedIds[k];
            if (id >= 0 && id < static_cast<int>(boardCorners.size()))
            {
                imagePointsForFrame.push_back(detectedCorners[k]);
                objectPointsForFrame.push_back(boardCorners[id]);
            }
        }
        if (imagePointsForFrame.size() >= 4)
        {
            imagePoints.push_back(std::move(imagePointsForFrame));
            objectPoints.push_back(std::move(objectPointsForFrame));
        }
    }
    if (imagePoints.size() >= 4)
        reprojectionError = cv::calibrateCamera(objectPoints, imagePoints, m_calibrationImageSize,
            cameraMatrix, distCoefficients, rotationVectors,
            translationVectors, calibrationFlags);
#else
    std::vector<std::vector<cv::Point2f>> allCornersConcatenated;
    std::vector<int> allIdsConcatenated;
    std::vector<int> markerCounterPerFrame;
    for (size_t i = 0; i < m_allCorners.size(); ++i)
    {
        markerCounterPerFrame.push_back(static_cast<int>(m_allCorners[i].size()));
        for (size_t j = 0; j < m_allCorners[i].size(); ++j)
        {
            allCornersConcatenated.push_back(m_allCorners[i][j]);
            allIdsConcatenated.push_back(m_allIds[i][j]);
        }
    }
    cv::aruco::calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated, markerCounterPerFrame,
        board, m_calibrationImageSize, cameraMatrix, distCoefficients,
        cv::noArray(), cv::noArray(), calibrationFlags);
    std::vector<cv::Mat> allCharucoCorners, allCharucoIds;
    for (size_t i = 0; i < m_allImages.size(); ++i)
    {
        cv::Mat corners, ids;
        cv::aruco::interpolateCornersCharuco(m_allCorners[i], m_allIds[i], m_allImages[i], board,
            corners, ids, cameraMatrix, distCoefficients);
        if (!corners.empty() && !ids.empty())
        {
            allCharucoCorners.push_back(corners);
            allCharucoIds.push_back(ids);
        }
    }
    if (allCharucoCorners.size() >= 4)
        reprojectionError = cv::aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, board,
            m_calibrationImageSize, cameraMatrix,
            distCoefficients, rotationVectors,
            translationVectors, calibrationFlags);
#endif

    if (!cameraMatrix.empty() && !distCoefficients.empty() && reprojectionError >= 0.0)
        finishCalibration(cameraMatrix, distCoefficients, m_calibrationImageSize, aspectRatio,
            calibrationFlags, reprojectionError);
}
