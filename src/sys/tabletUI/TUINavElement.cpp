/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include "TUINavElement.h"
#include "TUIApplication.h"
#include <QFrame>
#include <QMouseEvent>
#include <stdio.h>
#include <net/tokenbuffer.h>

class InputFrame : public QFrame
{
public:
    InputFrame(int id, QWidget *parent = 0);
    virtual ~InputFrame();

protected:
    virtual void mousePressEvent(QMouseEvent *);
    virtual void mouseReleaseEvent(QMouseEvent *);
    virtual void mouseMoveEvent(QMouseEvent *);

private:
    bool down; // true if mouse down
    int ID;
};
InputFrame::InputFrame(int id, QWidget *parent)
    : QFrame(parent)
{
    setBackgroundRole(QPalette::Base); // white background
    setAutoFillBackground(true);
    setEnabled(true);
    down = false;
    ID = id;
}

InputFrame::~InputFrame()
{
}

//
// Handles mouse press events for the connect widget.
//

void InputFrame::mousePressEvent(QMouseEvent *e)
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_PRESSED;
    tb << e->pos().x();
    tb << e->pos().y();
    TUIMainWindow::getInstance()->send(tb);
}

//
// Handles mouse release events for the connect widget.
//

void InputFrame::mouseReleaseEvent(QMouseEvent *e)
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_RELEASED;
    tb << e->pos().x();
    tb << e->pos().y();
    TUIMainWindow::getInstance()->send(tb);
}

//
// Handles mouse move events for the connect widget.
//

void InputFrame::mouseMoveEvent(QMouseEvent *e)
{

    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_POS;
    tb << e->pos().x();
    tb << e->pos().y();
    TUIMainWindow::getInstance()->send(tb);
}

/// Constructor
TUINavElement::TUINavElement(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    label = name;

    InputFrame *frame = new InputFrame(id, w);
    frame->setFrameStyle(QFrame::NoFrame);
    frame->setContentsMargins(5, 5, 5, 5);
#ifdef TABLET_PLUGIN
    frame->setMinimumHeight(140);
    frame->setMaximumHeight(140);
#else
    frame->setMinimumHeight(200);
    frame->setMaximumHeight(200);
#endif
    frame->setMinimumWidth(200);
    frame->setMaximumWidth(200);
    frame->setMouseTracking(true);
    widget = frame;
}

/// Destructor
TUINavElement::~TUINavElement()
{
    delete widget;
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUINavElement::setEnabled(bool en)
{
    (void)en;
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUINavElement::setHighlighted(bool hl)
{
    (void)hl;
}

const char *TUINavElement::getClassName() const
{
    return "TUINavElement";
}
