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

#include "TUISlider.h"
#include "TUIApplication.h"
#include "TUIContainer.h"

/// Constructor
TUISlider::TUISlider(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    width = 2;

    min = 0;
    max = 0;
    value = 0;
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

    widgets.insert(string);
    widgets.insert(slider);
}

/// Destructor
TUISlider::~TUISlider()
{
    delete string;
    delete slider;
    delete label;
}

void TUISlider::setPos(int x, int y)
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

const char *TUISlider::getClassName() const
{
    return "TUISlider";
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

void TUISlider::setLabel(QString textl)
{
    TUIElement::setLabel(textl);
    if (!label)
    {
        label = new QLabel(slider->parentWidget());
        widgets.insert(label);
        label->setBuddy(string);
        static_cast<QGridLayout *>(layout)->addWidget(label, 0, 0);
    }
    label->setText(textl);
}
