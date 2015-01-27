/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_MESSAGE_H_
#define _CUI_MESSAGE_H_

// OpenSceneGraph:
#include <osg/Vec3>
#include <osg/Geode>
#include <osgText/Text>

#include "Widget.h"

class vvStopwatch;

namespace cui
{

/** This class displays a text message on screen, for a limited time.
*/
class CUIEXPORT Message
{
public:
    enum AlignmentType
    {
        LEFT,
        CENTER,
        RIGHT
    };
    Message(AlignmentType);
    virtual ~Message();
    virtual void update();
    virtual void setText(std::string &, float = -1.0f);
    virtual void setText(const char *, float = -1.0f);
    virtual void setPosition(osg::Vec3 &);
    virtual osg::Node *getNode();
    virtual void setSize(float);
    virtual void setColor(osg::Vec4 &);

protected:
    osgText::Text *_messageText;
    osg::Geode *_geode;
    float _timeLeft; // time left for message display; -1 = always displayed
    vvStopwatch *_watch;
};
}

#endif
