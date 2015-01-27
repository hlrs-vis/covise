/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// OSG:
#include <osg/Geode>

// Virvo:
#include <virvo/vvclock.h>

// Local:
#include "Message.h"

using namespace osg;
using namespace cui;

Message::Message(Message::AlignmentType cuiAlign)
{
    _messageText = new osgText::Text;
    _messageText->setDataVariance(Object::DYNAMIC);
    _messageText->setFont(osgText::readFontFile("arial.ttf"));
    Vec4 color(1, 1, 1, 1);
    _messageText->setColor(color);
    _messageText->setFontResolution(20, 20);
    _messageText->setCharacterSize(0.2);
    osgText::Text::AlignmentType osgAlign;
    switch (cuiAlign)
    {
    default:
        osgAlign = osgText::Text::CENTER_CENTER;
        break;
    case LEFT:
        osgAlign = osgText::Text::LEFT_CENTER;
        break;
    case RIGHT:
        osgAlign = osgText::Text::RIGHT_CENTER;
        break;
    }
    _messageText->setAlignment(osgAlign);

    _geode = new Geode();
    _geode->addDrawable(_messageText);
    _geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    _geode->setNodeMask(~1);

    _timeLeft = 0;
    _watch = new vvStopwatch();
    _watch->start();
}

Message::~Message()
{
    delete _watch;
}

void Message::setPosition(Vec3 &pos)
{
    _messageText->setPosition(pos);
}

/** Call this to display a new message.
  @param duration display duration [seconds]; negative value for indefinite
*/
void Message::setText(std::string &text, float duration)
{
    _messageText->setText(text);
    _timeLeft = duration;
    _watch->start();
}

void Message::setText(const char *text, float duration)
{
    std::string string;
    string = text;
    setText(string, duration);
}

/// Call this in every draw callback:
void Message::update()
{
    if (_timeLeft > 0.0f) // is display time left?
    {
        _timeLeft -= _watch->getTime();

        _watch->start();

        if (_timeLeft <= 0.0)
        {
            _messageText->setText("");
            _timeLeft = 0.0f;
        }
    }
}

Node *Message::getNode()
{
    return (Node *)_geode;
}

void Message::setSize(float size)
{
    _messageText->setCharacterSize(size);
}

void Message::setColor(Vec4 &color)
{
    _messageText->setColor(color);
}
