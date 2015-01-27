/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TUIUIWidgetSet.h"

#include <climits>
#include <QAbstractButton>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QComboBox>
#include <QSlider>

#include <cassert>
#include <iostream>

TUIUIWidgetSet::TUIUIWidgetSet(QWidget *value, QWidget *control,
                               QList<QWidget *> views, QObject *parent)
    : QObject(parent)
    , commitWidget(control)
    , viewWidgets(views)
    , blockCommit(false)
{

    valueWidgets << value;

    if (value->property("parameterMin").isValid())
        this->helpers.insert("min", value->property("parameterMin"));

    if (value->property("parameterMax").isValid())
        this->helpers.insert("max", value->property("parameterMax"));

    this->type = value->property("parameterType").toString();

    assert(!this->type.endsWith("Vector"));

    initValue(valueWidgets[0]);
    initCommit();
    initViews();
}

TUIUIWidgetSet::TUIUIWidgetSet(QList<QWidget *> value, QWidget *control,
                               QList<QWidget *> views, QObject *parent)
    : QObject(parent)
    , valueWidgets(value)
    , commitWidget(control)
    , viewWidgets(views)
    , blockCommit(false)
{

    this->type = value[0]->property("parameterType").toString();

    assert(this->type.endsWith("Vector"));

    foreach (QWidget *v, this->valueWidgets)
        initValue(v);

    initCommit();
    initViews();
}

void TUIUIWidgetSet::initValue(QWidget *widget)
{

    QAbstractButton *button = qobject_cast<QAbstractButton *>(widget);
    QLineEdit *edit = qobject_cast<QLineEdit *>(widget);
    QListWidget *list = qobject_cast<QListWidget *>(widget);
    QComboBox *cb = qobject_cast<QComboBox *>(widget);
    QSlider *slider = qobject_cast<QSlider *>(widget);

    if (button != 0 && button->isCheckable())
        connect(button, SIGNAL(toggled(bool)),
                this, SLOT(checkableValueChanged(bool)));

    if (edit != 0)
        connect(edit, SIGNAL(textChanged(QString)),
                this, SLOT(textValueChanged(QString)));

    if (list != 0)
    {

        list->setSelectionMode(QAbstractItemView::SingleSelection);

        connect(list, SIGNAL(itemSelectionChanged()),
                this, SLOT(itemSelectionChanged()));
    }

    if (cb != 0)
    {
        connect(cb, SIGNAL(activated(QString)),
                this, SLOT(textValueChanged(QString)));
        connect(cb, SIGNAL(editTextChanged(QString)),
                this, SLOT(textValueChanged(QString)));
    }

    if (slider != 0)
    {
        if (this->type == "FloatSlider")
        {
            slider->setMinimum(0);
            slider->setMaximum(INT_MAX);
        }

        slider->setTracking(false);

        connect(slider, SIGNAL(valueChanged(int)),
                this, SLOT(sliderValueChanged(int)));
    }
}

void TUIUIWidgetSet::initCommit()
{

    QWidget *widget = this->commitWidget;

    QAbstractButton *button = qobject_cast<QAbstractButton *>(widget);
    //   QLineEdit *       edit    = qobject_cast<QLineEdit      *>(widget);
    //   QListWidget *     list    = qobject_cast<QListWidget    *>(widget);
    //   QComboBox *       cb      = qobject_cast<QComboBox      *>(widget);
    //   QSlider *         slider  = qobject_cast<QSlider        *>(widget);

    if (button != 0)
        connect(button, SIGNAL(clicked()),
                this, SLOT(commit()));
}

void TUIUIWidgetSet::initViews()
{

    foreach (QWidget *view, this->viewWidgets)
    {
        QLineEdit *edit = qobject_cast<QLineEdit *>(view);
        if (edit != 0)
        {
            edit->setReadOnly(true);
            ;
        }
    }
}

void TUIUIWidgetSet::setValue(const QVariant &value, const QList<QVariant> &helpers)
{

    bool valueChanged = this->value != value;

    this->value = value;

    this->blockCommit = true;

    if (this->type.endsWith("Slider"))
    {
        if (helpers.size() == 2)
        {
            if (this->helpers["min"] != helpers[0])
            {
                this->helpers["min"] = helpers[0];
                valueChanged = true;
            }
            if (this->helpers["max"] != helpers[1])
            {
                this->helpers["max"] = helpers[1];
                valueChanged = true;
            }
        }
    }

    if (!valueChanged)
    {
        this->blockCommit = false;
        return;
    }

    QList<QVariant> valueVector;

    if (this->type.endsWith("Vector"))
        valueVector = value.toList();
    else
        valueVector << this->value;

    for (int ctr = 0; ctr < valueWidgets.size(); ++ctr)
    {
        QWidget *widget = this->valueWidgets[ctr];
        int parameterIndex = 0;
        if (this->type.endsWith("Vector"))
            parameterIndex = widget->property("parameterIndex").toInt();

        QAbstractButton *button = qobject_cast<QAbstractButton *>(widget);
        QLineEdit *edit = qobject_cast<QLineEdit *>(widget);
        QListWidget *list = qobject_cast<QListWidget *>(widget);
        QComboBox *cb = qobject_cast<QComboBox *>(widget);
        QSlider *slider = qobject_cast<QSlider *>(widget);

        if (button != 0 && button->isCheckable())
            button->setChecked(valueVector[parameterIndex].toBool());

        if (edit != 0)
            edit->setText(valueVector[parameterIndex].toString());

        if (list != 0)
        {
            QList<QListWidgetItem *> items = list->findItems(valueVector[parameterIndex].toString(), Qt::MatchExactly);
            if (items.empty())
            {
                foreach (QListWidgetItem *i, list->selectedItems())
                {
                    i->setSelected(false);
                }
            }
            else
            {
                items.at(0)->setSelected(true);
            }
        }

        if (cb != 0)
        {
            int index = cb->findText(valueVector[parameterIndex].toString());
            if (index >= 0)
                cb->setCurrentIndex(index);
        }

        if (slider != 0)
        {
            int ival = valueVector[parameterIndex].toInt();
            if (this->type == "FloatSlider")
            {
                double minimum = this->helpers["min"].toDouble();
                double maximum = this->helpers["max"].toDouble();
                double ff = (value.toDouble() - minimum) / (maximum - minimum);
                ival = int(INT_MAX * ff);
                ival = int(INT_MAX * ff);
            }
            slider->setValue(ival);
        }
    }

    this->blockCommit = false;
}

QVariant TUIUIWidgetSet::getValue() const
{
    return this->value;
}

void TUIUIWidgetSet::checkableValueChanged(bool value)
{

    QVariant v(value);

    QString name = sender()->property("parameterName").toString() + "_" + sender()->property("parameterIndex").toString();

    // Update views
    foreach (QWidget *view, this->viewWidgets)
    {

        if (!view->objectName().contains(name))
            continue;

        QAbstractButton *button = qobject_cast<QAbstractButton *>(view);
        if (button != 0 && button->isCheckable())
        {
            button->setChecked(value);
            continue;
        }
        QLabel *label = qobject_cast<QLabel *>(view);
        if (label != 0)
        {
            label->setText(v.toString());
            continue;
        }
        QLineEdit *edit = qobject_cast<QLineEdit *>(view);
        if (edit != 0)
        {
            edit->setText(v.toString());
        }
    }

    if (this->type.endsWith("Vector"))
        v = this->value.toList().at(sender()->property("parameterIndex").toInt());
    else
        v = this->value;

    if (v.toBool() == value)
        return;

    if (this->type.endsWith("Vector"))
    {
        QList<QVariant> v2 = this->value.toList();
        v2[sender()->property("parameterIndex").toInt()] = value;
        this->value = v2;
    }
    else
        this->value = value;

    if (this->commitWidget == 0)
        commit();
}

void TUIUIWidgetSet::textValueChanged(QString value)
{

    QString name = sender()->property("parameterName").toString() + "_" + sender()->property("parameterIndex").toString();

    // Update views
    foreach (QWidget *view, this->viewWidgets)
    {

        if (!view->objectName().contains(name))
            continue;

        QLabel *label = qobject_cast<QLabel *>(view);
        if (label != 0)
        {
            label->setText(value);
            continue;
        }
        QLineEdit *edit = qobject_cast<QLineEdit *>(view);
        if (edit != 0)
        {
            edit->setText(value);
        }
    }

    if (this->type.endsWith("Vector"))
    {
        QList<QVariant> v2 = this->value.toList();
        if (v2[sender()->property("parameterIndex").toInt()] == value)
            return;
        v2[sender()->property("parameterIndex").toInt()] = value;
        this->value = v2;
    }
    else
    {
        if (this->value == value)
            return;
        this->value = value;
    }

    if (this->commitWidget == 0)
        commit();
}

void TUIUIWidgetSet::sliderValueChanged(int v)
{
    QVariant value;

    QString name = sender()->property("parameterName").toString() + "_" + sender()->property("parameterIndex").toString();

    if (this->type == "FloatSlider")
    {
#if QT_VERSION >= 0x040600
        float minimum = this->helpers["min"].toFloat();
        float maximum = this->helpers["max"].toFloat();
#else
        float minimum = this->helpers["min"].toDouble();
        float maximum = this->helpers["max"].toDouble();
#endif
        float fv = minimum + (float(v) / INT_MAX * (maximum - minimum));
        value = fv;
    }
    else
    {
        value = v;
    }

    // Update views
    foreach (QWidget *view, this->viewWidgets)
    {

        if (!view->objectName().contains(name))
            continue;

        QLabel *label = qobject_cast<QLabel *>(view);
        if (label != 0)
        {
            label->setText(value.toString());
            continue;
        }
        QLineEdit *edit = qobject_cast<QLineEdit *>(view);
        if (edit != 0)
        {
            edit->setText(value.toString());
        }
    }

    if (this->type.endsWith("Vector"))
    {
        QList<QVariant> v2 = this->value.toList();
        if (v2[sender()->property("parameterIndex").toInt()] == value)
            return;
        v2[sender()->property("parameterIndex").toInt()] = value;
        this->value = v2;
    }
    else
    {
        if (this->value == value)
            return;
        this->value = value;
    }

    if (this->commitWidget == 0)
        commit();
}

void TUIUIWidgetSet::itemSelectionChanged()
{
    QListWidget *list = qobject_cast<QListWidget *>(sender());
    QList<QListWidgetItem *> items = list->selectedItems();
    if (items.empty())
        textValueChanged("");
    else
        textValueChanged(items.at(0)->text());
}

void TUIUIWidgetSet::commit()
{
    if (blockCommit)
        return;

    QString value = this->value.toString();

    if (this->type.endsWith("Vector"))
    {
        QList<QVariant> values = this->value.toList();
        value = values[0].toString();
        for (int ctr = 1; ctr < values.length(); ++ctr)
            value += " " + values[ctr].toString();
    }

    QVariant scriptProperty;
    if (this->commitWidget)
        scriptProperty = this->commitWidget->property("COVERScript");
    else
        scriptProperty = this->valueWidgets[0]->property("COVERScript");

    if (scriptProperty.isValid())
    {
        QString script = scriptProperty.toString();

        if (script.contains("$") && this->type.endsWith("Vector"))
            std::cerr << "TUIUIWidgetSet::commit warn: calling function with vector type may give undesireable results" << std::endl;

        //TODO Do I need index substitution {$i} here
        script.replace("$", value);
        std::cerr << "TUIUIWidgetSet::commit info: found attached script: " << std::endl << qPrintable(script) << std::endl;
        emit evaluateScript(script);
    }
    else
    {
        emit parameterChanged(this->valueWidgets[0]->property("moduleID").toString(),
                              this->valueWidgets[0]->property("parameterName").toString(),
                              this->valueWidgets[0]->property("parameterType").toString(),
                              value,
                              this->valueWidgets[0]->property("parameterLinked").toBool());

        std::cerr << "TUIUIWidgetSet::commit info: no attached script" << std::endl;
    }
}
