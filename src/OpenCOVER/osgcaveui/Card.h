/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_CARD_H_
#define _CUI_CARD_H_

// C++:
#include <string>
#include <time.h>

// OSG:
#include <osg/Geometry>
#include <osg/Switch>

// Local:
#include "Widget.h"
#include "Events.h"

namespace cui
{
class Interaction;
class CardListener;
class InputDevice;

/** This is an abstract class for all button-like widgets.
   */
class CUIEXPORT Card : public Widget, public Events
{
public:
    Card(Interaction *);
    virtual ~Card();
    virtual void setText(const std::string &);
    virtual std::string getText();
    virtual bool loadImage(const std::string &);
    virtual bool loadCUIImage(const std::string &);
    virtual void setIconImage(osg::Image *);
    virtual void setDepthTest(bool);
    virtual void addCardListener(CardListener *);
    virtual float getWidth();
    virtual float getHeight();
    virtual void setFocus(bool);
    virtual void setHighlighted(bool);
    static void setMagnificationMode(bool);
    virtual void setSize(float);
    void setTipText(char[], bool = false);
    void setTipVisibility(bool);
    void setTipSize(float, float);
    void enableTipDelay(bool);
    void setTipDelay(double);

    static const float DEFAULT_CARD_WIDTH;
    static const float DEFAULT_CARD_HEIGHT;

protected:
    static const float DEFAULT_FONT_SIZE;
    static const float ICON_SIZE; ///< fraction of card width to use for icon
    static const float TIP_WIDTH;
    static bool _magnification; ///< true = magnification mode on
    osg::ref_ptr<osg::Switch> _swHighlight; ///< switches geometries for highlighted and non-highlighted
    osg::ref_ptr<osg::Switch> _swFocus; ///< switches geometries for focus and no focus
    osg::Geode *_highlight[2]; ///< highlight dependent geometry: 0=no highlight, 1=highlight
    osg::Geode *_focus[2]; ///< focus dependent geometry: 0=no focus, 1=in focus
    osgText::Text *_labelText[2]; ///< two instances: one for light, one for dark
    osg::Texture2D *_icon; ///< image on card
    Interaction *_interaction;
    std::list<CardListener *> _listeners;
    osg::Vec3 _magnifyOffset; ///< translation for magnified widgets to be in front of other widgets
    bool _cursorInside;
    ///< scale and translate for magnification mode
    osg::ref_ptr<osg::MatrixTransform> _magnifyXF;

    virtual void createGeometry();
    virtual osgText::Text *createLabel(const osg::Vec4 &, int);
    virtual osg::Geometry *createBackground(const osg::Vec4 &);
    virtual osg::Geometry *createIcon();
    virtual osg::Geometry *createFrame(const osg::Vec4 &);
    virtual void cursorEnter(InputDevice *);
    virtual void cursorUpdate(InputDevice *);
    virtual void cursorLeave(InputDevice *);
    virtual void buttonEvent(InputDevice *, int);
    virtual void joystickEvent(InputDevice *);
    virtual void wheelEvent(InputDevice *, int);

    // tooltip part
    osg::Geode *_tipGeode;
    osg::Vec3Array *_tipGeomVertices;
    osg::Vec3Array *_tipFrameVertices;
    osgText::Text *_tipString;
    osg::Vec3 _tipStringPos;
    float _tipWidth, _tipHeight;
    bool _tipVisible;
    bool _useTipDelay;
    double _tipDelay;
    time_t _enterTime;

    void createTip();
    void showTip(bool);
};

class CardListener
{
public:
    virtual ~CardListener()
    {
    }
    /** @param card
          @param button
          @param state
          @return true if event code handles event completely so it shouldn't do anything else
      */
    virtual bool cardButtonEvent(Card *, int, int) = 0;
    /** @param card
          @param inputDevice
          @return true if event code handles event completely so it shouldn't do anything else
      */
    virtual bool cardCursorUpdate(Card *, InputDevice *) = 0;
};
}
#endif
