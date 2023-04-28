#include "MatrixUtil.h"
#include <opencv2/calib3d.hpp>

osg::Matrix OpenGLToOSGMatrix;
osg::Matrix OSGToOpenGLMatrix;

osg::Vec3d toOsg(const cv::Vec3d &v)
{
    osg::Vec3d osgV{v[0], v[1], v[2]};
    return osgV *OpenGLToOSGMatrix;
}

cv::Vec3d toCv(const osg::Vec3d &v)
{
    auto cvV = v *OSGToOpenGLMatrix;
    return cv::Vec3d{cvV[0], cvV[1], cvV[2]};
}

osg::Matrix cvToOsgMat(const cv::Vec3d &rot, const cv::Vec3d &trans)
{
    cv::Mat markerRotMat(3, 3, CV_64F);
    setIdentity(markerRotMat);
    Rodrigues(rot, markerRotMat);
    
    // transform matrix
    double markerTransformData[16];
    cv::Mat markerTransformMat(4, 4, CV_64F, markerTransformData);
    setIdentity(markerTransformMat);
                
    // copy rot matrix to transform matrix
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            markerTransformMat.at<double>(i, j) = markerRotMat.at<double>(i, j);
        }
        // copy trans vector to transform matrix
        markerTransformMat.at<double>(i, 3) = trans[i];
    }

    osg::Matrix markerTrans;
    for (int u = 0; u < 4; u++)
        for (int v = 0; v < 4; v++)
            markerTrans(v, u) = markerTransformData[(u * 4) + v];

    return OSGToOpenGLMatrix * markerTrans * OpenGLToOSGMatrix;
}

cv::Vec3d getCvTranslation(const osg::Matrix &mat)
{
    auto v = mat.getTrans() * OSGToOpenGLMatrix;
    return cv::Vec3d{v[0], v[1], v[2]};
}

cv::Vec3d getCvRotation(const osg::Matrix &mat)
{
    cv::Mat cvMat(3,3, CV_64F);
    auto m = OpenGLToOSGMatrix *mat *OSGToOpenGLMatrix;

    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            cvMat.at<double>(i,j) = m(i,j);
        
    cv::Vec3d v;
    cv::Rodrigues(cvMat, v);
    return -v;
}