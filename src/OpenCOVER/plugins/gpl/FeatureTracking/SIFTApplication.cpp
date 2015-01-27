/*************************************************************************
// Plugin: FeatureTracking
// Description: Calculation of corresponding 2D points with SIFT features
// Date: 2010-07-01
// Author: RTW
//***********************************************************************/

#include "SIFTApplication.h"

#include "Features/FeatureEntry.h"
#include "SIFTFeatureEntry.h"
#include "Image/ImageProcessor.h"
#include "Image/PrimitivesDrawer.h"
#include "Math/Math2d.h"
#include <time.h>

SIFTApplication::SIFTApplication()
{
    matchesVector = new std::vector<CorrPoint>;
    numOfMatches = 0;
    debugMode = false;
}

SIFTApplication::~SIFTApplication()
{
}

void SIFTApplication::resetSIFTApplication()
{
    matchesVector->clear();
    numOfMatches = 0;
    debugMode = false;
}

const int SIFTApplication::findKeypoints(TrackingObject *inTrackObj)
{
    siftKeypointCalculator.SetThreshold(inTrackObj->getQualityThreshold());
    siftKeypointCalculator.SetNumberOfOctaves(1);
    inTrackObj->getKeypointArray()->Clear();
    const int numOfKeypoints = siftKeypointCalculator.CalculateFeatures(inTrackObj->getImage(), inTrackObj->getKeypointArray());
    inTrackObj->setNumberOfKeypoints(numOfKeypoints);
    matchesVector->reserve(numOfKeypoints);

    if (buildKdTree(inTrackObj))
    {
        return numOfKeypoints;
    }
    else
    {
        return -1;
    }
}

const int SIFTApplication::findMatches(TrackingObject *inTrackObj_R, TrackingObject *inTrackObj_C)
{
    // exclude keypoints in boundary area
    const float boundaryVal = 0.08f;
    const int imgWidth = inTrackObj_R->getImage()->width;
    const int imgHeight = inTrackObj_R->getImage()->height;
    const float boundaryWA = imgWidth * boundaryVal;
    const float boundaryHA = imgHeight * boundaryVal;
    const float boundaryWB = imgWidth - boundaryWA;
    const float boundaryHB = imgHeight - boundaryHA;

    // visualize boundary box
    if (debugMode)
    {
        const Vec2d boundaryLL = { boundaryWA, boundaryHA };
        const Vec2d boundaryLR = { boundaryWB, boundaryHA };
        const Vec2d boundaryUL = { boundaryWA, boundaryHB };
        const Vec2d boundaryUR = { boundaryWB, boundaryHB };
        PrimitivesDrawer::DrawLine(inTrackObj_R->getImage(), boundaryLR, boundaryLL, 255, 255, 255);
        PrimitivesDrawer::DrawLine(inTrackObj_R->getImage(), boundaryUR, boundaryUL, 255, 255, 255);
        PrimitivesDrawer::DrawLine(inTrackObj_R->getImage(), boundaryUL, boundaryLL, 255, 255, 255);
        PrimitivesDrawer::DrawLine(inTrackObj_R->getImage(), boundaryUR, boundaryLR, 255, 255, 255);
        PrimitivesDrawer::DrawLine(inTrackObj_C->getImage(), boundaryLR, boundaryLL, 255, 255, 255);
        PrimitivesDrawer::DrawLine(inTrackObj_C->getImage(), boundaryUR, boundaryUL, 255, 255, 255);
        PrimitivesDrawer::DrawLine(inTrackObj_C->getImage(), boundaryUL, boundaryLL, 255, 255, 255);
        PrimitivesDrawer::DrawLine(inTrackObj_C->getImage(), boundaryUR, boundaryLR, 255, 255, 255);
    }

    float *data = NULL;
    float error = 0.0;

    for (int i = 0; i < inTrackObj_C->getKeypointArray()->GetSize(); i++)
    {
        const CSIFTFeatureEntry *keypoint = (const CSIFTFeatureEntry *)inTrackObj_C->getKeypointArray()->GetElement(i);
        // determine matches according to the keypoints found in the reference image
        inTrackObj_R->getKdTree()->NearestNeighbour_HL_BBF_CC(keypoint->m_pFeature, inTrackObj_C->getKdLeaves(), error, data);
        const CSIFTFeatureEntry *matchedKeypoint = ((const CSIFTFeatureEntry *)((unsigned int *)data)[128]);

        if (error < inTrackObj_C->getMatchingThreshold())
        {
            const Vec2d &keypointVec = keypoint->point;
            const Vec2d &matchedVec = matchedKeypoint->point;

            // store keypoint matches
            if ((matchedVec.x > boundaryWA) && (matchedVec.y > boundaryHA) && (matchedVec.x < boundaryWB) && (matchedVec.y < boundaryHB))
            {
                CorrPoint match;
                match.setFirstX(matchedVec.x);
                match.setFirstY(matchedVec.y);
                match.setSecondX(keypointVec.x);
                match.setSecondY(keypointVec.y);
                matchesVector->push_back(match);
            }
        }
    }
    if (data != NULL)
    {
        delete data;
    }

    numOfMatches = matchesVector->size();

    // visualize single keypoints and keypoint matches
    if (debugMode)
    {
        drawKeypoints(matchesVector, inTrackObj_R, inTrackObj_C);
    }
    return numOfMatches;
}

bool SIFTApplication::buildKdTree(TrackingObject *inTrackObj)
{
    const int allKeypoints = inTrackObj->getKeypointArray()->GetSize();

    if (allKeypoints >= 1)
    {
        const int dimension = ((CSIFTFeatureEntry *)inTrackObj->getKeypointArray()->GetElement(0))->GetSize();
        const int overallDimension = dimension + 1;
        int i;
        float **values = new float *[allKeypoints];

        // build values for search tree generation
        for (i = 0; i < allKeypoints; i++)
        {
            const CSIFTFeatureEntry *keypoint = (CSIFTFeatureEntry *)inTrackObj->getKeypointArray()->GetElement(i);
            values[i] = new float[overallDimension];
            for (int j = 0; j < dimension; j++)
            {
                values[i][j] = keypoint->m_pFeature[j];
            }
            // set user data (only on 32 bit systems!)
            const unsigned int keypointPointer = (unsigned int)keypoint;
            memcpy(&values[i][dimension], &keypointPointer, 4);
        }
        inTrackObj->getKdTree()->Dump(false);
        inTrackObj->getKdTree()->Build(values, 0, allKeypoints - 1, 3, dimension, 2);

        for (i = 0; i < allKeypoints; i++)
        {
            delete[] values[i];
        }
        delete[] values;
        return true;
    }
    else
    {
        return false;
    }
}

bool SIFTApplication::convertImage(unsigned char *inInputImagePtr, TrackingObject *inTrackObj)
{
    if ((imgWidth == inTrackObj->getImage()->width) && (imgHeight == inTrackObj->getImage()->height))
    {
        const int numOfPixels = inTrackObj->getImage()->width * inTrackObj->getImage()->height;
        inTrackObj->getImage()->type = CByteImage::eGrayScale;

        for (int offset = 0, i = 0; i < numOfPixels; i++, offset += 3)
        {
            inTrackObj->getImage()->pixels[i] = (inInputImagePtr[offset] + (inInputImagePtr[offset + 1] << 1) + inInputImagePtr[offset + 2] + 2) >> 2;
        }
        return true;
    }
    else
    {
        return false;
    }
}

void SIFTApplication::setImageSize(int inImgWidth, int inImgHeight)
{
    imgWidth = inImgWidth;
    imgHeight = inImgHeight;
}

void SIFTApplication::drawKeypoints(std::vector<CorrPoint> *inMatchVector, TrackingObject *inTrackObj_R, TrackingObject *inTrackObj_C)
{
    // draw reference keypoints
    for (int i = 0; i < inTrackObj_R->getKeypointArray()->GetSize(); i++)
    {
        const CSIFTFeatureEntry *keypoint = (const CSIFTFeatureEntry *)inTrackObj_R->getKeypointArray()->GetElement(i);
        PrimitivesDrawer::DrawCross(inTrackObj_R->getImage(), keypoint->point, 3.0f, 255, 255, 255);
        PrimitivesDrawer::DrawCircle(inTrackObj_R->getImage(), keypoint->point.x, keypoint->point.y, 3.0f, 255, 255, 255);
    }

    // draw capture keypoints
    for (int i = 0; i < inTrackObj_C->getKeypointArray()->GetSize(); i++)
    {
        const CSIFTFeatureEntry *keypoint = (const CSIFTFeatureEntry *)inTrackObj_C->getKeypointArray()->GetElement(i);
        PrimitivesDrawer::DrawCross(inTrackObj_C->getImage(), keypoint->point, 3.0f, 255, 255, 255);
        PrimitivesDrawer::DrawCircle(inTrackObj_C->getImage(), keypoint->point.x, keypoint->point.y, 3.0f, 255, 255, 255);
    }

    // draw keypoint matches
    for (int i = 0; i < inMatchVector->size(); i++)
    {
        PrimitivesDrawer::DrawCircle(inTrackObj_R->getImage(), inMatchVector->at(i).getFirstX(), inMatchVector->at(i).getFirstY(), 3.5f, 255, 255, 255);
        PrimitivesDrawer::DrawCircle(inTrackObj_C->getImage(), inMatchVector->at(i).getSecondX(), inMatchVector->at(i).getSecondY(), 3.5f, 255, 255, 255);
        const Vec2d matchFirst = { inMatchVector->at(i).getFirstX(), inMatchVector->at(i).getFirstY() };
        const Vec2d matchSecond = { inMatchVector->at(i).getSecondX(), inMatchVector->at(i).getSecondY() };
        const Vec2d matchPt1 = { matchFirst.x + inTrackObj_R->getImage()->width, matchFirst.y };
        const Vec2d matchPt2 = { matchSecond.x - inTrackObj_C->getImage()->width, matchSecond.y };
        PrimitivesDrawer::DrawLine(inTrackObj_R->getImage(), matchFirst, matchPt1, 255, 255, 255);
        PrimitivesDrawer::DrawLine(inTrackObj_C->getImage(), matchSecond, matchPt2, 255, 255, 255);
    }
}

void SIFTApplication::setDebugMode(bool inMode)
{
    debugMode = inMode;
}

const int SIFTApplication::getNumberOfMatches()
{
    return (const int)numOfMatches;
}

std::vector<CorrPoint> *SIFTApplication::getMatchesVector()
{
    return matchesVector;
}

// EOF
