#ifndef COVER_PLUGIN_GHOSTAVATAR_UTIL_SanitizeRigidTransform_H
#define COVER_PLUGIN_GHOSTAVATAR_UTIL_SanitizeRigidTransform_H

#include <osg/Matrix>

/*
    Please note that these methods have been copied from src/OpenCOVER/PluginUtil/coVR3DTransformInteractor.cpp.
    TODO: make both classes use a common utilty header for these methods to avoid code duplication.
*/

double determinant3x3(const osg::Matrix &m);

osg::Matrix extractRotation(const osg::Matrix &m);

osg::Matrix sanitizeRigidTransform(const osg::Matrix &m);

#endif // COVER_PLUGIN_GHOSTAVATAR_UTIL_SanitizeRigidTransform_H