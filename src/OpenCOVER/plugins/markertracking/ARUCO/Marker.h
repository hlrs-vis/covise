#ifndef COVISE_ARUCO_MARKER_H
#define COVISE_ARUCO_MARKER_H

#include <array>
#include <utility>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <osg/Matrix>
#include <mutex>
namespace opencover{
class MarkerTrackingMarker;
}

class ArucoMarker
{
public:
    ArucoMarker(const opencover::MarkerTrackingMarker *arToolKitMarker);
    ArucoMarker(const ArucoMarker&) = delete;
    ArucoMarker(ArucoMarker&&) = default;
    ArucoMarker &operator=(const ArucoMarker&) = delete;
    ArucoMarker &operator=(ArucoMarker&&) = default;
    ~ArucoMarker() = default;
    int getCapturedAt(const std::vector<int> captureIDs); // returns -1 if not captured
    cv::Vec3d &cameraRot(int captureIdx);
    cv::Vec3d &cameraTrans(int captureIdx);
    void setCamera(const cv::Vec3d &cameraRot, const cv::Vec3d &cameraTrans, int captureIdx);
    const opencover::MarkerTrackingMarker *markerTrackingMarker;
    const std::array<cv::Vec3d, 4> corners;
    int capturedAt = -1;
    int lastCaptureIndex = 0;
    const int markerId = -1; //patternId converted to int
private:
    std::array<cv::Vec3d, 3> m_cameraRot, m_cameraTrans; //per capture index
    std::unique_ptr<std::mutex> m_mutex;
};


typedef std::vector<ArucoMarker> MultiMarker;
typedef std::vector<ArucoMarker*> MultiMarkerPtr;

const ArucoMarker *findMarker(const std::vector<MultiMarker> &multiMarkers, const opencover::MarkerTrackingMarker *marker);
const ArucoMarker *findMarker(const std::vector<MultiMarker> &multiMarkers, int id);

#endif //COVISE_ARUCO_MARKER_H