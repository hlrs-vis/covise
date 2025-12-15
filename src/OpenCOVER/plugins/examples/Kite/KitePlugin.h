/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef KITE_PLUGIN_H
#define KITE_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <osg/MatrixTransform>
#include <string>

class KitePlugin : public opencover::coVRPlugin
{
public:
    KitePlugin();
    ~KitePlugin() override;

    bool init() override;

private:
    bool loadModel(const std::string &path);

    osg::ref_ptr<osg::MatrixTransform> m_transform;
    osg::ref_ptr<osg::Node> m_model;
};

#endif // KITE_PLUGIN_H
