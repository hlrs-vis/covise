/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <assert.h>

#include "TUIApplication.h"
#include "TUIColorWidget.h"
#include <QLabel>
#include <QColor>
#include <QLineEdit>
#include <QSlider>
#include <QValidator>
#include <QGridLayout>

//Constructor
TUIColorWidget::TUIColorWidget(QWidget *parent)
    : QFrame(parent)
    , color(255, 255, 255, 255)
{
    this->setFrameStyle(QFrame::NoFrame);

    QHBoxLayout *mainLayout = new QHBoxLayout(this);
    QGridLayout *controlsLayout = new QGridLayout();

    redSlider = new EditSlider(0, 255, 1, color.red(), this, "R");
    greenSlider = new EditSlider(0, 255, 1, color.green(), this, "G");
    blueSlider = new EditSlider(0, 255, 1, color.blue(), this, "B");
    hueSlider = new EditSlider(0, 360, 1, color.hue(), this, "H");
    saturationSlider = new EditSlider(0, 255, 1, color.saturation(), this, "S");
    valueSlider = new EditSlider(0, 255, 1, color.value(), this, "V");
    //alphaSlider = new EditSlider(0,255,1,color.alpha(),this, "Alpha");

    controlsLayout->addWidget(redSlider, 1, 0, Qt::AlignCenter);
    controlsLayout->addWidget(greenSlider, 2, 0, Qt::AlignCenter);
    controlsLayout->addWidget(blueSlider, 3, 0, Qt::AlignCenter);
    //controlsLayout->addWidget(alphaSlider,4,0,Qt::AlignCenter);
    controlsLayout->addWidget(hueSlider, 1, 1, Qt::AlignCenter);
    controlsLayout->addWidget(saturationSlider, 2, 1, Qt::AlignCenter);
    controlsLayout->addWidget(valueSlider, 3, 1, Qt::AlignCenter);

    colorTriangle = new ColorTriangle(this);
    mainLayout->addWidget(colorTriangle);
    mainLayout->addLayout(controlsLayout);

    setLayout(mainLayout);

    connect(colorTriangle, SIGNAL(colorChanged(const QColor &)), this, SLOT(changeTriangle(const QColor &)));
    connect(colorTriangle, SIGNAL(released(const QColor &)), this, SLOT(changeTriangle(const QColor &)));

    connect(redSlider, SIGNAL(moved(int)), this, SLOT(changeRed(int)));
    connect(greenSlider, SIGNAL(moved(int)), this, SLOT(changeGreen(int)));
    connect(blueSlider, SIGNAL(moved(int)), this, SLOT(changeBlue(int)));
    connect(hueSlider, SIGNAL(moved(int)), this, SLOT(changeHue(int)));
    connect(saturationSlider, SIGNAL(moved(int)), this, SLOT(changeSaturation(int)));
    connect(valueSlider, SIGNAL(moved(int)), this, SLOT(changeValue(int)));
    //connect(alphaSlider, SIGNAL(moved(int)), this, SLOT(changeAlpha(int)));

    //external signals
    //connect(alphaSlider, SIGNAL(moved(int)), this, SIGNAL(changedAlpha(int)));
}

TUIColorWidget::~TUIColorWidget()
{
}

void TUIColorWidget::updateControls()
{
    colorTriangle->setColor(color);
    redSlider->setValue(color.red());
    greenSlider->setValue(color.green());
    blueSlider->setValue(color.blue());
    hueSlider->setValue(color.hue());
    saturationSlider->setValue(color.saturation());
    valueSlider->setValue(color.value());
    //alphaSlider->setValue(color.alpha());

    emit changedColor(color);
}

void TUIColorWidget::changeTriangle(const QColor &col)
{
    if (color != col)
    {
        color = col;
        updateControls();
    }
}

void TUIColorWidget::setColor(const QColor &col, int a)
{
    color = col;
    color.setAlpha(a);

    updateControls();
}

void TUIColorWidget::changeAlpha(int a)
{
    color.setAlpha(a);
    emit changedAlpha(a);
}

void TUIColorWidget::changeRed(int r)
{
    color.setRed(r);
    updateControls();
}

void TUIColorWidget::changeBlue(int b)
{
    color.setBlue(b);
    updateControls();
}

void TUIColorWidget::changeGreen(int g)
{
    color.setGreen(g);
    updateControls();
}

void TUIColorWidget::changeHue(int h)
{
    color.setHsv(h, color.saturation(), color.value(), color.alpha());
    updateControls();
}

void TUIColorWidget::changeSaturation(int s)
{
    color.setHsv(color.hue(), s, color.value(), color.alpha());
    updateControls();
}

void TUIColorWidget::changeValue(int v)
{
    color.setHsv(color.hue(), color.saturation(), v, color.alpha());
    updateControls();
}

EditSlider::EditSlider(int min, int max, int step, int start, QWidget *parent, const QString &name)
    : QGroupBox(parent)
{
    val = start;
    edit = new QLineEdit(QString("%1").arg(start), this);
    slider = new QSlider(Qt::Horizontal, this);
    slider->setRange(min, max);
    slider->setSingleStep(step);
    slider->setValue(start);

    label = new QLabel(this);
    label->setText(name);
    //label->setMinimumWidth(40);

    connect(slider, SIGNAL(sliderMoved(int)), this, SLOT(moveSlot(int)));
    connect(edit, SIGNAL(returnPressed()), this, SLOT(editChanged()));
    connect(edit, SIGNAL(textChanged(const QString &)), this, SLOT(editTextChanged(const QString &)));
    intValidator = new QIntValidator(min, max, this);
    edit->setValidator(intValidator);
    slider->setMinimumWidth(40);
    edit->setFixedWidth(30);

    QHBoxLayout *layout = new QHBoxLayout;
    layout->addWidget(label);
    layout->addWidget(edit);
    layout->addWidget(slider);
    setLayout(layout);
    setContentsMargins(0, 0, 0, 0);
}

void EditSlider::moveSlot(int value)
{
    if (val != value)
    {
        val = value;
        edit->setText(QString("%1").arg(val));
        emit moved(val);
    }
}

void EditSlider::setValue(int value)
{
    if (val != value)
    {
        val = value;
        edit->setText(QString("%1").arg(val));
        slider->setValue(val);
    }
}

void EditSlider::editChanged()
{
    bool ok;
    int value = edit->text().toInt(&ok, 10);
    if (ok)
    {
        if (value >= slider->minimum() && value <= slider->maximum() && value != val)
        {
            val = value;
            slider->setValue(val);
            emit moved(val);
        }
        else
        {
            edit->setText(QString("%1").arg(val));
        }
    }
}

void EditSlider::editTextChanged(const QString &text)
{
    (void)text;
}
