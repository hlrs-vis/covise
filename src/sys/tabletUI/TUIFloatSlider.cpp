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

const float SliderMax = 1000.f;
const float SliderTiny = 1e-15f;

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
    slider->setMaximum(SliderMax);

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
    if (this->ival == ival)
        return;
    this->ival = ival;

    if (logScale)
    {
        float lmin = std::log10(std::max(SliderTiny, min));
        float lmax = std::log10(std::max(SliderTiny, max));
        float lval = lmax * (ival/SliderMax) + lmin * ((SliderMax-ival)/SliderMax);
        value = std::pow(10.f, lval);
    }
    else
    {
        float delta = (max - min) / SliderMax;
        value = min + (delta * ival);
    }

    QString tmp;
    tmp = QString("%1").arg(value);
    string->setText(tmp);
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_MOVED;
    tb << value;
    TUIMainWindow::getInstance()->send(tb);
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
    showSliderValue(min, max, value);
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
    showSliderValue(min, max, value);
    //TUIMainWindow::getInstance()->getStatusBar()->message(QString("Floatslider: %1").arg(value));
}

const char *TUIFloatSlider::getClassName() const
{
    return "TUIFloatSlider";
}

void TUIFloatSlider::showSliderValue(float min, float max, float value)
{
    int oival = ival;
    if (value > max)
        value = max;
    if (value < min)
        value = min;
    if ((max - min) == 0.0)
    {
        ival = 0;
    }
    else
    {
        if (logScale)
        {
            float lmin = std::log10(std::max(SliderTiny, min));
            float lmax = std::log10(std::max(SliderTiny, max));
            float lval = std::log10(std::max(SliderTiny, value));
            lval = lmin + ((lmax - lmin) * ((int)(SliderMax * ((lval - lmin) / (lmax - lmin)))) / SliderMax);
            ival = (int)(0.5f+(SliderMax * ((lval - lmin) / (lmax - lmin))));
            lval = lmin + ((lmax - lmin)/SliderMax*ival);
            value = std::pow(10.f, lval);
        }
        else
        {
            ival = (int)(0.5f+(SliderMax * ((value - min) / (max - min))));
            value = min + ((max - min)/SliderMax*ival);
        }
    }
    if (ival != oival)
    {
        slider->setValue(ival);
    }
    QString tmp;
    tmp = QString("%1").arg(value);
    string->setText(tmp);
}

void TUIFloatSlider::setValue(TabletValue type, covise::TokenBuffer &tb)
{
    //cerr << "TUIFloatSlider::setValue " << name.toStdString()<< ": type = " << type << endl;
    if (type == TABLET_MIN)
    {
        tb >> min;
        showSliderValue(min, max, value);
    }
    else if (type == TABLET_MAX)
    {
        tb >> max;
        showSliderValue(min, max, value);
    }
    else if (type == TABLET_FLOAT)
    {
        tb >> value;
        showSliderValue(min, max, value);
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
    else if (type == TABLET_SLIDER_SCALE)
    {
        int scale;
        tb >> scale;
        logScale = scale==TABLET_SLIDER_LOGARITHMIC;
        showSliderValue(min, max, value);
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
