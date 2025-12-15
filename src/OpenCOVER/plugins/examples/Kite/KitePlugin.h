/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef KITE_PLUGIN_H
#define KITE_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <osg/MatrixTransform>
#include <string>
#include <vector>
#include <osg/Vec3>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/LineWidth>

class KitePlugin : public opencover::coVRPlugin
{
public:
    KitePlugin();
    ~KitePlugin() override;

    bool init() override;
    void preFrame() override;

private:
    bool loadModel(const std::string &path);
    void parseCsv(const std::string &path);
    void updateTransform(int frameIndex);

    struct Frame
    {
        double t = 0.0;
        osg::Vec3 pos;
        double roll = 0.0;
        double pitch = 0.0;
        double yaw = 0.0;
        double tetherLength = 0.0;
        double tetherForce = 0.0;
        double tetherReeloutSpeed = 0.0;
        double windSpeed = 0.0;
        double windDir = 0.0;
        double kiteElevation = 0.0;
        double kiteAzimuth = 0.0;
        double kiteHeading = 0.0;
        double kiteCourse = 0.0;
    };

    osg::ref_ptr<osg::MatrixTransform> m_transform;
    osg::ref_ptr<osg::Node> m_model;
    osg::ref_ptr<osg::Geode> m_tetherGeode;
    osg::ref_ptr<osg::Geometry> m_tetherGeom;
    osg::ref_ptr<osg::LineWidth> m_tetherWidth;
    osg::ref_ptr<osg::Geode> m_groundGeode;
    osg::ref_ptr<osg::Geode> m_windGeode;
    osg::ref_ptr<osg::Geometry> m_windGeom;
    std::vector<Frame> m_frames;
    std::string m_csvPath;
    double m_maxForce = 0.0;
};

#endif // KITE_PLUGIN_H
