/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include "qtcolortriangle.h"
#include "TUIColorTriangle.h"
#include "TUIApplication.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

/// Constructor
TUIColorTriangle::TUIColorTriangle(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    colorTriangle = new QtColorTriangle(w);

    widget = colorTriangle;

    connect(widget, SIGNAL(colorChanged(const QColor &)), this, SLOT(changeColor(const QColor &)));
    connect(widget, SIGNAL(released(const QColor &)), this, SLOT(releaseColor(const QColor &)));
}

/// Destructor
TUIColorTriangle::~TUIColorTriangle()
{
    delete widget;
}

void TUIColorTriangle::changeColor(const QColor &col)
{
    red = ((float)col.red()) / 255.0;
    green = ((float)col.green()) / 255.0;
    blue = ((float)col.blue()) / 255.0;
    alpha = ((float)col.alpha()) / 255.0;
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_RGBA;
    tb << TABLET_PRESSED;
    tb << red;
    tb << green;
    tb << blue;
    tb << alpha;

    TUIMainWindow::getInstance()->send(tb);
}

void TUIColorTriangle::releaseColor(const QColor &col)
{
    red = ((float)col.red()) / 255.0;
    green = ((float)col.green()) / 255.0;
    blue = ((float)col.blue()) / 255.0;
    alpha = ((float)col.alpha()) / 255.0;
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_RGBA;
    tb << TABLET_RELEASED;
    tb << red;
    tb << green;
    tb << blue;
    tb << alpha;

    TUIMainWindow::getInstance()->send(tb);
}

void TUIColorTriangle::setValue(int type, covise::TokenBuffer &tb)
{

    if (type == TABLET_RED)
    {
        tb >> red;
    }
    else if (type == TABLET_GREEN)
    {
        tb >> green;
    }
    else if (type == TABLET_BLUE)
    {
        tb >> blue;
    }
    int r = (int)(red * 255);
    int g = (int)(green * 255);
    int b = (int)(blue * 255);
    int a = (int)(alpha * 255);

    /*if(r>255) r = 255;
   if(g>255) g = 255;
   if(b>255) b = 255;
   if(r<0) r = 0;
   if(g<0) g = 0;
   if(b<0) b = 0;*/

    QColor col(r, g, b, a);

    colorTriangle->setColor(col);

    TUIElement::setValue(type, tb);
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIColorTriangle::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIColorTriangle::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

char *TUIColorTriangle::getClassName()
{
    return (char *)"TUIColorTriangle";
}

bool TUIColorTriangle::isOfClassName(char *classname)
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return TUIElement::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
