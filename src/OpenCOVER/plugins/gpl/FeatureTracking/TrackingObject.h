/****************************************************************************
// Plugin: FeatureTrackingPlugin
// Description: Estimation of camera pose from corresponding 2D image points
// Date: 2010-06-11
// Author: RTW
//***************************************************************************/

#ifndef _TRACKINGOBJECT_H
#define _TRACKINGOBJECT_H

#include "SIFTApplication.h"
#include <osg/Matrix>

class SIFTApplication;

class TrackingObject
{
public:
    void initTrackingObject(int inWidth, int inHeight);
    void setKeypointArray(CDynamicArray *inKeypointArray);
    void setKdTree(CKdTree *inKdTree);
    void setNumberOfKeypoints(int inNumberOfKeypoints);
    void setQualityThreshold(float inQualityThreshold);
    void setMatchingThreshold(float inMatchingThreshold);
    void setKdLeaves(int inKdLeaves);
    void setCameraPosition(osg::Matrix inMat);
    void setID(int inFrame);
    void setFitValue(int inFitVal);

    CByteImage *getImage();
    CDynamicArray *getKeypointArray();
    CKdTree *getKdTree();
    int getNumberOfKeypoints();
    float getQualityThreshold();
    float getMatchingThreshold();
    int getKdLeaves();
    osg::Matrix getCameraPose();
    int getID();
    int getFitValue();

protected:
    // SIFT features
    CByteImage *image;
    CDynamicArray *keypointArray;
    CKdTree *kdTree;
    int numberOfKeypoints;
    float qualityThreshold;
    float matchingThreshold;
    int kdLeaves;

    // absolute camera position as matrix
    osg::Matrix camPose;

    // general properties
    int id;
    // 0 = image conversion failed, -1 = not enough keypoints found, -2 = camera pose estimation failed
    int fit;
};

class CorrPoint
{
public:
    void setFirstX(float inFirstX);
    void setFirstY(float inFirstY);
    void setSecondX(float inSecondX);
    void setSecondY(float inSecondY);
    float getFirstX();
    float getFirstY();
    float getSecondX();
    float getSecondY();

protected:
    float firstX; // x coordinate of the first point correspondence
    float firstY; // y coordinate of the first point correspondence
    float secondX; // x coordinate of the second point correspondence
    float secondY; // y coordinate of the second point correspondence
};

inline void TrackingObject::initTrackingObject(int inWidth, int inHeight)
{
    image = new CByteImage(inWidth, inHeight, CByteImage::eRGB24);
    keypointArray = new CDynamicArray(3000);
    kdTree = new CKdTree();
    numberOfKeypoints = 0;
    qualityThreshold = 0.05f;
    matchingThreshold = 0.4f;
    kdLeaves = 50;
    camPose.makeIdentity();
    id = 0;
    fit = 1;
}

inline void TrackingObject::setKeypointArray(CDynamicArray *inKeypointArray)
{
    keypointArray = inKeypointArray;
}

inline void TrackingObject::setKdTree(CKdTree *inKdTree)
{
    kdTree = inKdTree;
}

inline void TrackingObject::setNumberOfKeypoints(int inNumberOfKeypoints)
{
    numberOfKeypoints = inNumberOfKeypoints;
}

inline void TrackingObject::setQualityThreshold(float inQualityThreshold)
{
    qualityThreshold = inQualityThreshold;
}

inline void TrackingObject::setMatchingThreshold(float inMatchingThreshold)
{
    matchingThreshold = inMatchingThreshold;
}

inline void TrackingObject::setKdLeaves(int inKdLeaves)
{
    kdLeaves = inKdLeaves;
}

inline void TrackingObject::setCameraPosition(osg::Matrix inMat)
{
    camPose = inMat;
}

inline void TrackingObject::setID(int inID)
{
    id = inID;
}

inline void TrackingObject::setFitValue(int inFitVal)
{
    fit = inFitVal;
}

inline void CorrPoint::setFirstX(float inFirstX)
{
    firstX = inFirstX;
}

inline void CorrPoint::setFirstY(float inFirstY)
{
    firstY = inFirstY;
}

inline void CorrPoint::setSecondX(float inSecondX)
{
    secondX = inSecondX;
}

inline void CorrPoint::setSecondY(float inSecondY)
{
    secondY = inSecondY;
}

inline CByteImage *TrackingObject::getImage()
{
    return image;
}

inline CDynamicArray *TrackingObject::getKeypointArray()
{
    return keypointArray;
}

inline CKdTree *TrackingObject::getKdTree()
{
    return kdTree;
}

inline int TrackingObject::getNumberOfKeypoints()
{
    return numberOfKeypoints;
}

inline float TrackingObject::getQualityThreshold()
{
    return qualityThreshold;
}

inline float TrackingObject::getMatchingThreshold()
{
    return matchingThreshold;
}

inline int TrackingObject::getKdLeaves()
{
    return kdLeaves;
}

inline osg::Matrix TrackingObject::getCameraPose()
{
    return camPose;
}

inline int TrackingObject::getID()
{
    return id;
}

inline int TrackingObject::getFitValue()
{
    return fit;
}

inline float CorrPoint::getFirstX()
{
    return firstX;
}

inline float CorrPoint::getFirstY()
{
    return firstY;
}

inline float CorrPoint::getSecondX()
{
    return secondX;
}

inline float CorrPoint::getSecondY()
{
    return secondY;
}

#endif

// EOF
