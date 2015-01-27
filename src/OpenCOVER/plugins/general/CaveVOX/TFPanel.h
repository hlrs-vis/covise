/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TFPANEL_H_
#define _TFPANEL_H_

// Inspace:
#include <osgDrawObj.H>

// OSG:
#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osgText/Text>
#include <osg/LineSegment>

// CUI:
#include "CheckBox.H"
#include "Interaction.H"
#include "Dial.H"
#include "Events.H"
#include "Card.H"
#include "TextureWidget.H"
#include "Panel.H"

// Virvo:
#include <vvtransfunc.h>
#include <vvtfwidget.h>

#include "TFColorWidget.H"
#include "TFColorBar.H"
#include "TFColorBox.H"

class osgDrawObj;

namespace cui
{
class TFListener;

class TFPanel : public cui::Panel, public BarListener, public cui::DialChangeListener
{
public:
    TFPanel(Interaction *, osgDrawObj *, Appearance, Movability = NON_MOVABLE);
    virtual ~TFPanel();
    virtual void cursorEnter(cui::InputDevice *);
    virtual void cursorUpdate(cui::InputDevice *);
    virtual void cursorLeave(cui::InputDevice *);
    virtual void buttonEvent(cui::InputDevice *, int);
    virtual void addTFListener(TFListener *);

    virtual void draw1DTF();
    virtual void drawPinBackground();
    virtual void drawColorTexture();
    virtual void drawPinLines();
    virtual void drawPinLine(vvTFWidget *w);

    virtual void handleSelection(vvTFWidget *);
    virtual void moveWidget(float);

    virtual bool cardButtonEvent(Card *, int, int);
    virtual bool cardCursorUpdate(Card *, InputDevice *);

    virtual void dialValueChanged(Dial *dial, float newValue);

    virtual void setColor(osg::Vec4);
    virtual void displayWheel(bool);

    virtual void setTF(vvTransFunc *);

protected:
    list<TFListener *> _TFListeners;
    cui::TFColorBox *_boxTexture;
    cui::TFColorBar *_barTexture;
    cui::TextureWidget *_selectionTexture;

    osg::Image *_boxImage;
    osg::Image *_barImage;

    cui::Button *_bGaussian;
    cui::Button *_bPyramid;
    cui::Button *_bColor;
    cui::Button *_bNewColor;
    cui::Button *_bDelete;
    cui::CheckBox *_cbSelectColor;
    cui::Button *_bOK;
    cui::Button *_bCancel;

    cui::Dial *_dTopWidth;
    cui::Dial *_dBottomWidth;

    int _selColor;

    vvTransFunc *_transFunc;

    vvTFWidget *_selectedWidget;

    list<TFColorWidget *> _widgets;
};
class TFListener
{
public:
    virtual void getNextColor(bool) = 0;
    virtual void setFunction(vvTransFunc *) = 0;
    virtual void setTFVisible(bool) = 0;
};
}
#endif
