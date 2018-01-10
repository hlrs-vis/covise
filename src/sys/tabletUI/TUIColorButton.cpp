/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include "qtcolortriangle.h"
#include "qpushbutton.h"
#include <QColorDialog>
#include "TUIColorButton.h"
#include "TUIApplication.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

/// Constructor
TUIColorButton::TUIColorButton(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    colorButton = new QPushButton(w);

    widget = colorButton;

    connect(colorButton, SIGNAL(pressed()), this, SLOT(onColorButtonPressed()));

    //connect(widget, SIGNAL(colorChanged(const QColor &)), this, SLOT(changeColor(const QColor &)));
    //connect(widget, SIGNAL(released(const QColor &)), this, SLOT(releaseColor(const QColor &)));
}

/// Destructor
TUIColorButton::~TUIColorButton()
{
    delete widget;
}

void TUIColorButton::onColorButtonPressed()
{
    QColor initC = colorButton->palette().color(QPalette::Button);
    QRgb my = QColorDialog::getRgba(initC.rgba(), 0, colorButton);
    int r = qRed(my);
    int g = qGreen(my);
    int b = qBlue(my);
    int alpha = qAlpha(my);
    QColor color = QColor(r, g, b, alpha);
    if (color.isValid())
    {
        changeColor(color);
    }
}

void TUIColorButton::changeColor(const QColor &col)
{
    colorButton->setPalette(QPalette(col));
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

void TUIColorButton::releaseColor(const QColor &col)
{
    red = ((float)col.red()) / 255.0;
    green = ((float)col.green()) / 255.0;
    blue = ((float)col.blue()) / 255.0;
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

void TUIColorButton::setValue(int type, covise::TokenBuffer &tb)
{

    if (type == TABLET_RGBA)
    {
        tb >> red;
        tb >> green;
        tb >> blue;
        tb >> alpha;
        int r = (int)(red * 255);
        int g = (int)(green * 255);
        int b = (int)(blue * 255);
        int a = (int)(alpha * 255);

        QColor col(r, g, b, a);
        colorButton->setPalette(QPalette(col));
    }

    TUIElement::setValue(type, tb);
}

const char *TUIColorButton::getClassName() const
{
    return "TUIColorButton";
}
