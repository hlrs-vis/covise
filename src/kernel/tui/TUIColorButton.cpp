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
#include "TUIMain.h"
#include <net/tokenbuffer.h>

/// Constructor
TUIColorButton::TUIColorButton(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    colorButton = createWidget<QPushButton>(w);

    connect(colorButton, SIGNAL(pressed()), this, SLOT(onColorButtonPressed()));

    //connect(widget, SIGNAL(colorChanged(const QColor &)), this, SLOT(changeColor(const QColor &)));
    //connect(widget, SIGNAL(released(const QColor &)), this, SLOT(releaseColor(const QColor &)));
}

void TUIColorButton::onColorButtonPressed()
{
    QColor initC = colorButton->palette().color(QPalette::Button);
    QColor my = QColorDialog::getColor(initC.rgba(), colorButton);
    int r = my.red();
    int g = my.green();
    int b = my.blue();
    int alpha = my.alpha();
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

    TUIMain::getInstance()->send(tb);
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

    TUIMain::getInstance()->send(tb);
}

void TUIColorButton::setValue(TabletValue type, covise::TokenBuffer &tb)
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
