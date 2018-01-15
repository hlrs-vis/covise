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
#include <QGridLayout>

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
    width = 2;

    min = 0.0;
    max = 0.0;
    value = 0.0;

    slider = new QSlider(w);

    string = new QLineEdit(w);

    connect(string, SIGNAL(returnPressed()), this, SLOT(released()));
    slider->setMinimum(0);
    slider->setMaximum(1000);

    connect(slider, SIGNAL(valueChanged(int)), this, SLOT(sliderChanged(int)));
    connect(slider, SIGNAL(sliderPressed()), this, SLOT(pressed()));
    connect(slider, SIGNAL(sliderReleased()), this, SLOT(released()));

    auto gl = new QGridLayout;
    layout = gl;
    gl->addWidget(slider, 1, 0, 1, width-1);
    gl->addWidget(string, 1, width-1);
    for (int i=0; i<width-1; ++i)
        gl->setColumnStretch(i, 100);
    gl->setContentsMargins(0, 0, 0, 0);

    widgets.insert(string);
    widgets.insert(slider);
}

/// Destructor
TUIFloatSlider::~TUIFloatSlider()
{
    delete layout;
    delete string;
    delete slider;
    delete label;
}

void TUIFloatSlider::setPos(int x, int y)
{
    xPos = x;
    yPos = y;
    TUIContainer *parent = getParent();
    if (parent)
    {
        parent->addElementToLayout(this);
    }
    else
    {
        TUIMainWindow::getInstance()->addElementToLayout(this);
    }
    slider->setVisible(!hidden);
    string->setVisible(!hidden);
    if (label)
        label->setVisible(!hidden);

    slider->setMinimumWidth((width-1)*string->width());
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

const char *TUIFloatSlider::getClassName() const
{
    return "TUIFloatSlider";
}

void TUIFloatSlider::setValue(TabletValue type, covise::TokenBuffer &tb)
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
    else if (type == TABLET_ORIENTATION)
    {
        int orientation;
        tb >> orientation;
        if (orientation == Qt::Vertical)
        {
            slider->setOrientation(Qt::Vertical);
        }
        else
        {
            slider->setOrientation(Qt::Horizontal);
        }
    }
    TUIElement::setValue(type, tb);
}

void TUIFloatSlider::setLabel(QString textl)
{
    TUIElement::setLabel(textl);
    if (textl.isEmpty())
    {
        widgets.erase(label);
        delete label;
        label = nullptr;
    }
    else if (!label)
    {
        label = new QLabel(slider->parentWidget());
        widgets.insert(label);
        label->setBuddy(string);
        static_cast<QGridLayout *>(layout)->addWidget(label, 0, 0);
    }
    if (label)
        label->setText(textl);
}
