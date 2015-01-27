/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PATH_H
#define _PATH_H

#include <osg/Geometry>
#include <osg/Geode>

#include <vector>

class Path : public osg::Group
{
public:
    Path();
    ~Path();

    void update(float animationSeconds);
    void updateTransform();
    void loadFromFile(std::string filename);
    osg::Vec3 getCurrentPosition();
    osg::Vec3 getCurrentOrientation();

    void changeTranslation(osg::Vec3 offset);
    void changeRotation(float offset);
    void changeStartDrawing(int offset);

private:
    void drawLine();

    osg::ref_ptr<osg::Geode> pathGeode;
    osg::ref_ptr<osg::Geometry> pathGeometry;

    std::vector<osg::Vec3> points;
    std::vector<osg::Vec3> positions;
    std::vector<osg::Vec3> orientations;
    int animationIndex;

    // settings
    bool data_lockZ;
    osg::Vec3 data_translation;
    osg::Vec3 data_rotationAxis;
    float data_rotationAngle;
    float data_scale;
    int data_startDrawing;
};

#endif
