/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TUIUIScriptWidget.h"

#include <QAbstractButton>
#include <QAbstractSlider>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QComboBox>
#include <QLineEdit>

#include <iostream>
#include <cassert>

#define WIDGET_CONNECT(x, y)                                     \
    {                                                            \
        x *widget = qobject_cast<x *>(parent);                   \
        if (widget)                                              \
        {                                                        \
            connect(widget, SIGNAL(y), this, SLOT(activated())); \
            return;                                              \
        }                                                        \
    }

TUIUIScriptWidget::TUIUIScriptWidget(QWidget *parent)
    : QObject(parent)
{
    WIDGET_CONNECT(QAbstractButton, clicked());
    WIDGET_CONNECT(QAbstractSlider, valueChanged(int));
    WIDGET_CONNECT(QSpinBox, valueChanged(QString));
    WIDGET_CONNECT(QDoubleSpinBox, valueChanged(QString));
    WIDGET_CONNECT(QComboBox, currentIndexChanged(QString));
    WIDGET_CONNECT(QLineEdit, editingFinished());
}

/// Activates a script. Replaces $ in the script with the widget value, {$i} with the current index.
void TUIUIScriptWidget::activated()
{

    // Event will be handled by TUIUIWidgetSet
    if (sender()->property("moduleID").isValid())
        return;

    QVariant value = sender()->property("value");
    if (!value.isValid())
        value = sender()->property("checked");
    if (!value.isValid())
        value = sender()->property("text");
    if (!value.isValid())
        value = sender()->property("currentText");
    if (!value.isValid())
    {
        std::cerr << "TUIUIScriptWidget::activated err: unsupported widget type " << sender()->metaObject()->className() << std::endl;
        return;
    }

    QVariant index = sender()->property("currentIndex");

    QVariant scriptProperty = sender()->property("COVERScript");
    assert(scriptProperty.isValid());

    QString script = scriptProperty.toString();

    if (index.isValid())
        script.replace("{$i}", index.toString());

    script.replace("$", value.toString());
    std::cerr << "TUIUIScriptWidget::activated info: found attached script: " << std::endl << qPrintable(script) << std::endl;

    emit scriptActivated(script);
}
