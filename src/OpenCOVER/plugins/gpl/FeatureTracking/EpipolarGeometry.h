/****************************************************************************
// Plugin: FeatureTrackingPlugin
// Description: Estimation of camera pose from corresponding 2D image points
// Date: 2010-07-01
// Author: RTW
//***************************************************************************/

#ifndef _EPIPOLARGEOMETRY_H
#define _EPIPOLARGEOMETRY_H

#include "SIFTApplication.h"

#include <cv.h>
#include <osg/Matrix>
#include <osg/Geometry>

typedef struct
{
    float focLen;
    float imgCtrX;
    float imgCtrY;
    float skew;
} CamIntrinsic;

class TrackingObject;

class EpipolarGeometry
{
public:
    EpipolarGeometry();
    ~EpipolarGeometry();

    // find transformation between two camera poses using point correspondences
    bool findCameraTransformation(std::vector<CorrPoint> *inMatchesVec, TrackingObject *inTrackObj_R, TrackingObject *inTrackObj_C);
    float calcDistanceScale(std::vector<osg::Vec3> *in3DPoints);
    void resetEpipolarGeo();

    void setFocalLength(float inFocalLen);
    void setImageCenter(float inImgCtrX, float inImgCtrY);
    void setSkewParameter(float inSkew);
    void setInitMode(bool inMode);
    void setDebugMode(bool inMode);

    const int getRateOfCorrectMatches();
    osg::Matrix getCameraTransform();
    float getCameraFocalLength();
    float getCameraImageCenterX();
    float getCameraImageCenterY();
    float getSkewParameter();

    // just for testing
    //bool testFundMat(std::vector<CorrPoint> *inMatchesVec);
    std::vector<CorrPoint> *do2DTrafo(int inOp);

private:
    // camera parameters
    CamIntrinsic camIntr;
    osg::Matrix camTransMat;
    const int correctMatchesRate;

    // tracking parameters
    std::vector<osg::Vec3> *points3DVec;
    float trlScale;
    unsigned int projMat;

    // control flags
    bool debugMode;
    bool isInitMode;

    // returns the percentage of correct matches
    int evaluateEpipolarMat33(CvMat *inMat, std::vector<CorrPoint> *inMatchesVec);
    std::vector<CorrPoint> *normalizeCoords(std::vector<CorrPoint> *inMatchesVec);
    void displayFloatMat(CvMat *inMat);
};

#endif

// EOF
