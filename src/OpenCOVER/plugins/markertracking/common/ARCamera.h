#ifndef COVISE_AR_CAMERA_H
#define COVISE_AR_CAMERA_H

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/videoio.hpp>

#if (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7) || (CV_VERSION_MAJOR >= 5)
#include <opencv2/objdetect/charuco_detector.hpp>
#else
#include <opencv2/aruco/charuco.hpp>
#endif

#if CV_VERSION_MAJOR < 5
#include <opencv2/aruco.hpp>
#endif

#include <string>
#include <vector>

class ARCamera
{
public:
	struct Device
	{
		int id;
		std::string label;
	};

	static std::vector<Device> availableDevices();
	void setCalibrationFilename(const std::string &filename);
	bool open(int deviceId);
	void close();
	bool isOpened() const;
	bool read(cv::Mat &frame);

	int width() const;
	int height() const;
	const cv::Mat &cameraMatrix() const;
	const cv::Mat &distortionCoefficients() const;
	bool hasIntrinsics() const;

	bool consumeDeferredCalibrationRequest();
	bool isCalibrating() const;
	void startCalibration();
	void calibrateFrame(const cv::Mat &frame,
						const std::vector<std::vector<cv::Point2f>> &markerCorners,
						const std::vector<int> &markerIds,
						const cv::Ptr<cv::aruco::CharucoDetector> &detector,
						const cv::Ptr<cv::aruco::CharucoBoard> &board);

private:
	bool loadCalibration(const cv::Mat &frame);
	bool saveCalibration(const cv::Size &imageSize, float aspectRatio, int flags,
						 const cv::Mat &cameraMatrix, const cv::Mat &distortionCoefficients,
						 double reprojectionError) const;
	void finishCalibration(const cv::Mat &cameraMatrix, const cv::Mat &distortionCoefficients,
						   const cv::Size &imageSize, float aspectRatio, int flags,
						   double reprojectionError);

	cv::VideoCapture m_capture;
	cv::Mat m_cameraMatrix;
	cv::Mat m_distortionCoefficients;
	std::string m_calibrationFilename;
	int m_width = 0;
	int m_height = 0;
	bool m_deferredCalibration = false;
	bool m_calibrating = false;
	double m_lastCalibrationCapture = 0.0;
	cv::Size m_calibrationImageSize;
	std::vector<std::vector<std::vector<cv::Point2f>>> m_allCorners;
	std::vector<std::vector<int>> m_allIds;
	std::vector<cv::Mat> m_allImages;
};

#endif
