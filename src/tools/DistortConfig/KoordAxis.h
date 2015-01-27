/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include <osgViewer/Viewer>
#include <osgText/Text>
#include <osg/LineWidth>
#include <osg/Geometry>
#include <osgGA/TrackballManipulator>

class KoordAxis
{
public:
    KoordAxis(void);
    ~KoordAxis(void);

    osgText::Text *createAxisLabel(const std::string &iLabel, const osg::Vec3 &iPosition);
    osg::Geometry *createArrow(const osg::Matrixd &iTransform, const osg::Vec4 &iColor, double iHeight);
    osg::Geometry *createXAxis(double iHeight);
    osg::Geometry *createYAxis(double iHeight);
    osg::Geometry *createZAxis(double iHeight);
    osg::Geode *createAxesGeometry(double length);
};
