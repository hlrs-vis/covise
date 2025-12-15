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
    };

    osg::ref_ptr<osg::MatrixTransform> m_transform;
    osg::ref_ptr<osg::Node> m_model;
    std::vector<Frame> m_frames;
    std::string m_csvPath;
};

#endif // KITE_PLUGIN_H
