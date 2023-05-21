#include "Marker.h"
#include <cover/MarkerTracking.h>
#include <opencv2/calib3d/calib3d.hpp>
#include "MatrixUtil.h"
std::array<cv::Vec3d, 4> getMarkerCorners(const opencover::MarkerTrackingMarker *arToolKitMarker)
{
    std::array<cv::Vec3d, 4> corners;
    auto size = arToolKitMarker->getSize();
    corners[3] = (cv::Vec3d(-size / 2.f, size / 2.f, 0));
    corners[2] = (cv::Vec3d(size / 2.f, size / 2.f, 0));
    corners[1] = (cv::Vec3d(size / 2.f, -size / 2.f, 0));
    corners[0] = (cv::Vec3d(-size / 2.f, -size / 2.f, 0));
    for (size_t i = 0; i < 4; i++)
    {
        auto c = toOsg(corners[i]);
        c = c * arToolKitMarker->getOffset();
        corners[i] = toCv(c);
    }
    return corners;
}

int convertPatternId(const std::string &id)
{
    try {
        return std::stoi(id);
    }
    catch(const std::exception&) {
        return -1;
    }
}

ArucoMarker::ArucoMarker(const opencover::MarkerTrackingMarker *marker)
: markerTrackingMarker(marker)
, corners(getMarkerCorners(marker))
, m_mutex(new std::mutex)
, markerId(convertPatternId(marker->getPattern()))
{}

int ArucoMarker::getCapturedAt(const std::vector<int> captureIDs)
{
    
    auto it = std::find(captureIDs.begin(), captureIDs.end(), markerId);
    capturedAt = it == captureIDs.end() ? -1 : std::distance(captureIDs.begin(), it);
    return capturedAt;
}

cv::Vec3d &ArucoMarker::cameraRot(int captureIdx)
{
    std::lock_guard<std::mutex> g(*m_mutex);
    return m_cameraRot[captureIdx];
}

cv::Vec3d &ArucoMarker::cameraTrans(int captureIdx)
{
    std::lock_guard<std::mutex> g(*m_mutex);
    return m_cameraTrans[captureIdx];
}

void ArucoMarker::setCamera(const cv::Vec3d &cameraRot, const cv::Vec3d &cameraTrans, int captureIdx)
{
    std::lock_guard<std::mutex> g(*m_mutex);
    m_cameraRot[captureIdx] = cameraRot;
    m_cameraTrans[captureIdx] = cameraTrans;

}

const ArucoMarker *findMarker(const std::vector<MultiMarker> &multiMarkers, const opencover::MarkerTrackingMarker *marker)
{
    for(const auto &multiMarker : multiMarkers)
    {
        for(const auto &arucoMarker : multiMarker)
        {
            if(arucoMarker.markerTrackingMarker == marker)
                return &arucoMarker;
        }
    }
    return nullptr;
}

const ArucoMarker *findMarker(const std::vector<MultiMarker> &multiMarkers, int id)
{
    for(const auto &multiMarker : multiMarkers)
    {
        for(const auto &arucoMarker : multiMarker)
        {
            if(arucoMarker.markerId == id)
                return &arucoMarker;
        }
    }
    return nullptr;
}