/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>
#include <iostream>

#include <QLabel>
#include <QSlider>
#include <QString>
#include <QLineEdit>

#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

#include "TUIFloatSlider.h"
#include "TUIApplication.h"
#include "TUIContainer.h"

/// Constructor
TUIFloatSlider::TUIFloatSlider(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    //int row  = 0;

    min = 0.0;
    max = 0.0;
    value = 0.0;

    slider = new QSlider(w);

    string = new QLineEdit(w);

    connect(string, SIGNAL(returnPressed()), this, SLOT(released()));
    slider->setMinimum(0);
    slider->setMaximum(1000);

    widget = slider;
    connect(slider, SIGNAL(valueChanged(int)), this, SLOT(sliderChanged(int)));
    connect(slider, SIGNAL(sliderPressed()), this, SLOT(pressed()));
    connect(slider, SIGNAL(sliderReleased()), this, SLOT(released()));
}

/// Destructor
TUIFloatSlider::~TUIFloatSlider()
{
    delete string;
    delete slider;
}

void TUIFloatSlider::setPos(int x, int y)
{
    xPos = x;
    yPos = y;
    TUIContainer *parent;
    widget = slider;
    if ((parent = getParent()))
    {
        parent->addElementToLayout(this);
    }
    else
    {
        TUIMainWindow::getInstance()->addElementToLayout(this);
    }
    xPos++;
    widget = string;
    if ((parent = getParent()))
    {
        parent->addElementToLayout(this);
    }
    else
    {
        TUIMainWindow::getInstance()->addElementToLayout(this);
    }
    xPos--;
    slider->setVisible(!hidden);
    string->setVisible(!hidden);
}

void TUIFloatSlider::sliderChanged(int ival)
{
    float delta = (max - min) / 1000.0;
    float newVal = min + (delta * ival);
    if ((newVal < value - (delta / 2.0)) || (newVal > value + (delta / 2.0)))
    {
        value = newVal;
        QString tmp;
        tmp = QString("%1").arg(value);
        string->setText(tmp);
        covise::TokenBuffer tb;
        tb << ID;
        tb << 10;
        tb << value;
        TUIMainWindow::getInstance()->send(tb);
    }
    //TUIMainWindow::getInstance()->getStatusBar()->message(QString("Floatslider: %1").arg(value));
}

void TUIFloatSlider::pressed()
{
    value = string->text().toFloat();
    //std::cerr << "val:" << value << std::endl;
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_PRESSED;
    tb << value;
    TUIMainWindow::getInstance()->send(tb);
    //TUIMainWindow::getInstance()->getStatusBar()->message(QString("Floatslider: %1").arg(value));
}

void TUIFloatSlider::released()
{
    value = string->text().toFloat();
    //std::cerr << "rval:" << value << std::endl;
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_RELEASED;
    tb << value;
    TUIMainWindow::getInstance()->send(tb);
    //TUIMainWindow::getInstance()->getStatusBar()->message(QString("Floatslider: %1").arg(value));
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIFloatSlider::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIFloatSlider::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

const char *TUIFloatSlider::getClassName() const
{
    return "TUIFloatSlider";
}

bool TUIFloatSlider::isOfClassName(const char *classname) const
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

void TUIFloatSlider::setValue(int type, covise::TokenBuffer &tb)
{
    //cerr << "TUIFloatSlider::setValue " << name.toStdString()<< ": type = " << type << endl;
    if (type == TABLET_MIN)
    {
        tb >> min;
        if (value > max)
            value = max;
        if (value < min)
            value = min;
        if ((max - min) == 0.0)
            slider->setValue(0);
        else
            slider->setValue((int)(1000.0 * ((value - min) / (max - min))));
        QString tmp;
        tmp = QString("%1").arg(value);
        string->setText(tmp);
    }
    else if (type == TABLET_MAX)
    {
        tb >> max;
        if (value > max)
            value = max;
        if (value < min)
            value = min;
        if ((max - min) == 0.0)
            slider->setValue(0);
        else
            slider->setValue((int)(1000.0 * ((value - min) / (max - min))));
        QString tmp;
        tmp = QString("%1").arg(value);
        string->setText(tmp);
    }
    else if (type == TABLET_FLOAT)
    {
        tb >> value;
        if (value > max)
            value = max;
        if (value < min)
            value = min;
        if ((max - min) == 0.0)
            slider->setValue(0);
        else
        {
            float oval = value;
            value = min + ((max - min) * ((int)(1000.0 * ((value - min) / (max - min)))) / 1000.0);
            slider->setValue((int)(1000.0 * ((oval - min) / (max - min))));
        }
        QString tmp;
        tmp = QString("%1").arg(value);
        string->setText(tmp);
    }
    else if (type == TABLET_BOOL)
    {
        char state;
        tb >> state;
        bool bState = (bool)state;
        if (bState)
            slider->setOrientation(Qt::Horizontal);
        else
            slider->setOrientation(Qt::Vertical);
    }
    TUIElement::setValue(type, tb);
}
