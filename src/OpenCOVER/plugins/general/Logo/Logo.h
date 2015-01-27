/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LOGO_H
#define _LOGO_H

#include <cover/coVRPlugin.h>
#include <util/common.h>

#include <osg/Camera>
#include <osg/Geode>

class Logo
{
public:
    Logo(std::string domain, osg::ref_ptr<osg::Camera> camera);
    ~Logo();

    void hide();
    void show();

    bool isValid();

private:
    osg::ref_ptr<osg::Geode> m_geode;
    osg::ref_ptr<osg::Camera> m_camera;

    bool m_isValid;
};
#endif
