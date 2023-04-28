#ifndef ARUCO_MATRIX_UTIL_H
#define ARUCO_MATRIX_UTIL_H
#include <osg/Matrix>
#include <opencv2/core/matx.hpp>

extern osg::Matrix OpenGLToOSGMatrix;
extern osg::Matrix OSGToOpenGLMatrix;

osg::Vec3d toOsg(const cv::Vec3d &v);

cv::Vec3d toCv(const osg::Vec3d &v);

osg::Matrix cvToOsgMat(const cv::Vec3d &rot, const cv::Vec3d &trans);

cv::Vec3d getCvTranslation(const osg::Matrix &mat);

cv::Vec3d getCvRotation(const osg::Matrix &mat);

#endif // ARUCO_MATRIX_UTIL_H