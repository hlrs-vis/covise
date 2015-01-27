/****************************************************************************
// Plugin: FeatureTrackingPlugin
// Description: Estimation of camera pose from corresponding 2D image points
// Date: 2010-05-27
// Author: RTW
//***************************************************************************/

#include "TrackingObject.h"

TrackingObject::TrackingObject()
{
    imageGray = NULL;
    keypointArray = new CDynamicArray(3000);
    kdTree = new CKdTree();
    numberOfKeypoints = 0;
    qualityThreshold = 0.05f;
    matchingThreshold = 0.4f;
    kdLeaves = 50;

    rotX = 0;
    rotY = 0;
    rotZ = 0;
    trlX = 0;
    trlY = 0;
    trlZ = 0;

    confidenceVal = 0.0;
}

TrackingObject::~TrackingObject()
{
}

void TrackingObject::initImage(int inWidth, int inHeight)
{
    imageGray = new CByteImage(inWidth, inHeight, CByteImage::eGrayScale);
}

void TrackingObject::setKeypointArray(CDynamicArray *inKeypointArray)
{
    keypointArray = inKeypointArray;
}

void TrackingObject::setKdTree(CKdTree *inKdTree)
{
    kdTree = inKdTree;
}

void TrackingObject::setNumberOfKeypoints(int inNumberOfKeypoints)
{
    numberOfKeypoints = inNumberOfKeypoints;
}

void TrackingObject::setQualityThreshold(float inQualityThreshold)
{
    qualityThreshold = inQualityThreshold;
}

void TrackingObject::setMatchingThreshold(float inMatchingThreshold)
{
    matchingThreshold = inMatchingThreshold;
}

void TrackingObject::setKdLeaves(int inKdLeaves)
{
    kdLeaves = inKdLeaves;
}

void TrackingObject::setCameraCoordinates(int inRotX, int inRotY, int inRotZ, int inTrlX, int inTrlY, int inTrlZ)
{
    rotX = inRotX;
    rotY = inRotY;
    rotZ = inRotZ;
    trlX = inTrlX;
    trlY = inTrlY;
    trlZ = inTrlZ;
}

void TrackingObject::setConfidenceValue(float inConfidVal)
{
    confidenceVal = inConfidVal;
}

CByteImage *TrackingObject::getImage()
{
    return imageGray;
}

CDynamicArray *TrackingObject::getKeypointArray()
{
    return keypointArray;
}

CKdTree *TrackingObject::getKdTree()
{
    return kdTree;
}

int TrackingObject::getNumberOfKeypoints()
{
    return numberOfKeypoints;
}

float TrackingObject::getQualityThreshold()
{
    return qualityThreshold;
}

float TrackingObject::getMatchingThreshold()
{
    return matchingThreshold;
}

int TrackingObject::getKdLeaves()
{
    return kdLeaves;
}

int TrackingObject::getCameraRotX()
{
    return rotX;
}

int TrackingObject::getCameraRotY()
{
    return rotY;
}

int TrackingObject::getCameraRotZ()
{
    return rotZ;
}

int TrackingObject::getCameraTrlX()
{
    return trlX;
}

int TrackingObject::getCameraTrlY()
{
    return trlY;
}

int TrackingObject::getCameraTrlZ()
{
    return trlZ;
}

float TrackingObject::getConfidenceValue()
{
    return confidenceVal;
}

// EOF
