/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <assert.h>
#include "TUITab.h"
#include "TUIApplication.h"
#include "TUIColorTab.h"
#include <QLabel>
#include <QColor>
#include <QLineEdit>
#include <QSlider>
#include <QValidator>
#include <QGridLayout>
#include <net/tokenbuffer.h>

//Constructor
TUIColorTab::TUIColorTab(int id, int type, QWidget *w, int parent, QString name)
    : TUITab(id, type, w, parent, name)
{

    label = name;

    frame = new QFrame(w);
    frame->setFrameStyle(QFrame::NoFrame);

    widget = frame;

    red = 255;
    green = 255;
    blue = 255;
    alpha = 255;
    rGBtoHSV();

    auto grid = new QGridLayout(frame);
    layout = grid;
    colorTriangle = new QtColorTriangle(frame);
    grid->addWidget(colorTriangle, 0, 0, 1, 2, Qt::AlignCenter);

    redSlider = new EditSlider(0, 255, 1, red, frame, "");
    greenSlider = new EditSlider(0, 255, 1, green, frame, "");
    blueSlider = new EditSlider(0, 255, 1, blue, frame, "");
    hueSlider = new EditSlider(0, 360, 1, hue, frame, "");
    saturationSlider = new EditSlider(0, 255, 1, saturation, frame, "");
    valueSlider = new EditSlider(0, 255, 1, value, frame, "");
    alphaSlider = new EditSlider(0, 255, 1, alpha, frame, "");

    grid->addWidget(redSlider, 1, 0, Qt::AlignCenter);
    grid->addWidget(greenSlider, 2, 0, Qt::AlignCenter);
    grid->addWidget(blueSlider, 3, 0, Qt::AlignCenter);
    grid->addWidget(alphaSlider, 4, 0, Qt::AlignCenter);
    grid->addWidget(hueSlider, 1, 1, Qt::AlignCenter);
    grid->addWidget(saturationSlider, 2, 1, Qt::AlignCenter);
    grid->addWidget(valueSlider, 3, 1, Qt::AlignCenter);

    connect(colorTriangle, SIGNAL(colorChanged(const QColor &)), this, SLOT(changedTriangle(const QColor &)));
    connect(colorTriangle, SIGNAL(released(const QColor &)), this, SLOT(changedTriangle(const QColor &)));

    connect(redSlider, SIGNAL(moved(int)), this, SLOT(changedRed(int)));
    connect(greenSlider, SIGNAL(moved(int)), this, SLOT(changedGreen(int)));
    connect(blueSlider, SIGNAL(moved(int)), this, SLOT(changedBlue(int)));
    connect(hueSlider, SIGNAL(moved(int)), this, SLOT(changedHue(int)));
    connect(saturationSlider, SIGNAL(moved(int)), this, SLOT(changedSaturation(int)));
    connect(valueSlider, SIGNAL(moved(int)), this, SLOT(changedValue(int)));
    connect(alphaSlider, SIGNAL(moved(int)), this, SLOT(changedAlpha(int)));
}

TUIColorTab::~TUIColorTab()
{
}

void TUIColorTab::changedTriangle(const QColor &col)
{
    red = col.red();
    green = col.green();
    blue = col.blue();

    rGBtoHSV();

    changeRedColor();
    changeGreenColor();
    changeBlueColor();
    changeHueColor();
    changeSaturationColor();
    changeValueColor();

    sendColor();
}

void TUIColorTab::changedAlpha(int a)
{
    alpha = a;
    sendColor();
}

void TUIColorTab::changedRed(int r)
{
    red = r;

    rGBtoHSV();

    changeTriangleColor();
    changeHueColor();
    changeSaturationColor();
    changeValueColor();

    sendColor();
}

void TUIColorTab::changedBlue(int b)
{
    blue = b;

    rGBtoHSV();

    changeTriangleColor();
    changeHueColor();
    changeSaturationColor();
    changeValueColor();

    sendColor();
}

void TUIColorTab::changedGreen(int g)
{
    green = g;

    rGBtoHSV();

    changeTriangleColor();
    changeHueColor();
    changeSaturationColor();
    changeValueColor();

    sendColor();
}

void TUIColorTab::changedHue(int h)
{
    hue = h;

    hSVtoRGB();

    changeTriangleColor();
    changeRedColor();
    changeGreenColor();
    changeBlueColor();

    sendColor();
}

void TUIColorTab::changedSaturation(int s)
{
    saturation = s;

    hSVtoRGB();

    changeTriangleColor();
    changeRedColor();
    changeGreenColor();
    changeBlueColor();

    sendColor();
}

void TUIColorTab::changedValue(int v)
{
    value = v;
    hSVtoRGB();
    changeTriangleColor();
    changeRedColor();
    changeGreenColor();
    changeBlueColor();

    sendColor();
}

void TUIColorTab::hSVtoRGB()
{
    QColor color(0, 0, 0);
    color.setHsv(hue, saturation, value);
    red = color.red();
    green = color.green();
    blue = color.blue();
}

void TUIColorTab::rGBtoHSV()
{
    QColor color(Qt::red, Qt::green, Qt::blue);
    color.getHsv(&hue, &saturation, &value);
}

void TUIColorTab::sendColor()
{
    float r = ((float)red) / 255.0;
    float g = ((float)green) / 255.0;
    float b = ((float)blue) / 255.0;
    float a = ((float)alpha) / 255.0;
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_RGBA;
    tb << r;
    tb << g;
    tb << b;
    tb << a;

    TUIMainWindow::getInstance()->send(tb);
}

void TUIColorTab::setValue(TabletValue type, covise::TokenBuffer &tb)
{

    if (type == TABLET_RGBA)
    {
        float r, g, b, a;
        tb >> r;
        tb >> g;
        tb >> b;
        tb >> a;

        red = int(r * 255.0f);
        green = int(g * 255.0f);
        blue = int(b * 255.0f);
        alpha = int(a * 255.0f);

        rGBtoHSV();

        changeTriangleColor();
        changeRedColor();
        changeGreenColor();
        changeBlueColor();
        changeHueColor();
        changeSaturationColor();
        changeValueColor();
    }

    TUIElement::setValue(type, tb);
}

const char *TUIColorTab::getClassName() const
{
    return "TUIColorTab";
}
