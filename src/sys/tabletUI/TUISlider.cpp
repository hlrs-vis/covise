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

#include "TUISlider.h"
#include "TUIApplication.h"
#include "TUIContainer.h"

/// Constructor
TUISlider::TUISlider(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    //int row  = 0;

    min = 0;
    max = 0;
    value = 0;
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
TUISlider::~TUISlider()
{
    delete string;
    delete slider;
}

void TUISlider::setPos(int x, int y)
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

void TUISlider::sliderChanged(int ival)
{
    if (ival != value)
    {
        value = ival;
        QString tmp;
        tmp = QString("%1").arg(value);
        string->setText(tmp);
        covise::TokenBuffer tb;
        tb << ID;
        tb << 10;
        tb << value;
        TUIMainWindow::getInstance()->send(tb);
    }
    //TUIMainWindow::getInstance()->getStatusBar()->message(QString("Slider: %1").arg(value));
}

void TUISlider::pressed()
{
    value = string->text().toInt();
    //std::cerr << "val:" << value << std::endl;
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_PRESSED;
    tb << value;
    TUIMainWindow::getInstance()->send(tb);
    //TUIMainWindow::getInstance()->getStatusBar()->message(QString("Slider: %1").arg(value));
}

void TUISlider::released()
{
    value = string->text().toInt();
    //std::cerr << "rval:" << value << std::endl;
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_RELEASED;
    tb << value;
    TUIMainWindow::getInstance()->send(tb);
    //TUIMainWindow::getInstance()->getStatusBar()->message(QString("Slider: %1").arg(value));
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUISlider::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUISlider::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

const char *TUISlider::getClassName() const
{
    return "TUISlider";
}

bool TUISlider::isOfClassName(const char *classname) const
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

void TUISlider::setValue(int type, covise::TokenBuffer &tb)
{
    //cerr << "TUISlider::setValue info: type = " << type << endl;
    if (type == TABLET_MIN)
    {
        tb >> min;
        if (value > max)
            value = max;
        if (value < min)
            value = min;
        slider->setMinimum(min);
        slider->setValue(value);
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
        slider->setMaximum(max);
        slider->setValue(value);
        QString tmp;
        tmp = QString("%1").arg(value);
        string->setText(tmp);
    }
    else if (type == TABLET_INT)
    {
        tb >> value;
        if (value > max)
            value = max;
        if (value < min)
            value = min;
        slider->setValue(value);
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
