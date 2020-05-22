/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_HUD_H
#define CO_HUD_H

/*! \file
 \brief  OpenCOVER startup splash

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2006
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   2006
 */

#include <util/common.h>
#include <osg/Version>

#include <osg/Camera>
#include <osg/Geode>
#include <osgText/Text>
namespace opencover
{
class COVEREXPORT coHud
{
    coHud();
public:
    virtual ~coHud();
    static coHud *instance();

    void show();
    void hide();
    void redraw();
    void hideLater(float time = -1.);
    bool update();
    bool isVisible() const;

    void setText1(const std::string &text);
    void setText2(const std::string &text);
    void setText3(const std::string &text);

protected:
    bool visible;
    osg::ref_ptr<osgText::Text> line1;
    osg::ref_ptr<osgText::Text> line2;
    osg::ref_ptr<osgText::Text> line3;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Camera> camera;

    double hudTime;
    double logoTime;
    bool doHide;

private:
    static coHud *instance_;
};
}
#endif
