/*************************************************************************
// Plugin: FeatureTracking
// Description: Calculation of corresponding 2D points with SIFT features
// Date: 2010-06-11
// Author: RTW
//***********************************************************************/

#ifndef _SIFTAPPLICATION_H
#define _SIFTAPPLICATION_H

#include "Image/ByteImage.h"
#include "DataStructures/DynamicArray.h"
#include "DataStructures/KdTree/KdTree.h"
#include "Features/SIFTFeatures/SIFTFeatureCalculator.h"
#include "TrackingObject.h"
#include "CorrPoint.h"

class TrackingObject;

class SIFTApplication
{
public:
    SIFTApplication();
    ~SIFTApplication();

    enum DrawType
    {
        Cross,
        Circle,
        Point,
        Line
    };

    void resetSIFTApplication();
    bool convertImage(unsigned char *inInputImagePtr, TrackingObject *inTrackObj);
    void setImageSize(int inImageWidth, int inImageHeight);
    // finds SIFT keypoints in a gray-scale CByteImage-image
    const int findKeypoints(TrackingObject *inTrackObj);
    // finds matches of keypoints on a gray-scale CByteImage-image, returns number of matches
    const int findMatches(TrackingObject *inTrackObj_R, TrackingObject *inTrackObj_C);

    void setDebugMode(bool inMode);

    // returns a vector with the found matches of keypoints
    std::vector<CorrPoint> *getMatchesVector();
    const int getNumberOfMatches();

private:
    CKdTree *kdTree;
    // keypoints
    CDynamicArray *keypointArray;
    // keypoint matches within a defined window of the image for visualization
    CDynamicArray *matchedKeypointArray;
    // keypoint matches within a defined window of the image
    std::vector<CorrPoint> *matchesVector;
    CSIFTFeatureCalculator siftKeypointCalculator;

    int numOfMatches;
    int imgWidth;
    int imgHeight;

    // control flags
    bool debugMode;

    bool buildKdTree(TrackingObject *inTrackObj);
    void drawKeypoints(std::vector<CorrPoint> *inMatchVector, TrackingObject *inTrackObj_A, TrackingObject *inTrackObj_B);
};

#endif

// EOF
