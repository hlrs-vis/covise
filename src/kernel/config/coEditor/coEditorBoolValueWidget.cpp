/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coEditorBoolValueWidget.h"
#include <QCheckBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QAction>

coEditorBoolValueWidget::coEditorBoolValueWidget(QWidget *parent, const QString name, Type type)
    : coEditorValueWidget(parent, type)
{
    widgetName = name;
}

coEditorBoolValueWidget::~coEditorBoolValueWidget()
{
}

void coEditorBoolValueWidget::setValue(const QString &valueName, const QString &value,
                                       const QString &readableAttrRule,
                                       const QString &attributeDescription,
                                       bool required, const QRegExp &rx)
{
    fvariable = valueName;
    fvalue = value;

    QString attributeRule = readableAttrRule;
    rx.isEmpty();

    QHBoxLayout *layout;
    QLabel *valueLabel;
    QSpacerItem *spacerItem;

    setObjectName(widgetName);
    setMaximumSize(QSize(16777215, 100));
    setBaseSize(QSize(400, 25));
    setToolTip(attributeDescription);

    layout = new QHBoxLayout(this);
    layout->setSpacing(0);
    layout->setMargin(0);
    layout->setObjectName(QString::fromUtf8("layout"));
    valueLabel = new QLabel(this);
    valueLabel->setText(valueName);
    valueLabel->setObjectName(QString::fromUtf8("valueLabel"));
    valueLabel->setBaseSize(QSize(150, 22));

    QSizePolicy labelSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
    labelSizePolicy.setHorizontalStretch(1);
    labelSizePolicy.setVerticalStretch(0);
    labelSizePolicy.setHeightForWidth(valueLabel->sizePolicy().hasHeightForWidth());
    valueLabel->setSizePolicy(labelSizePolicy);

    spacerItem = new QSpacerItem(30, 20, QSizePolicy::Preferred, QSizePolicy::Minimum);

    aValuesCheckBox = new QCheckBox(this);
    aValuesCheckBox->setObjectName(valueName);
    aValuesCheckBox->setToolTip(attributeDescription);
    aValuesCheckBox->setTristate(!required);
    aValuesCheckBox->setAutoFillBackground(1);

    //aValuesCheckBox-> setStyleSheet(" QCheckBox::indicator:unchecked {  image: url(:/images/apply.png);} ");

    //create Contextmenu - Delete value
    deleteValueAction = new QAction(tr("Delete this Value ?"), aValuesCheckBox);
    deleteValueAction->setStatusTip(tr("Delete this value ?"));
    addAction(deleteValueAction);
    setContextMenuPolicy(Qt::ActionsContextMenu);

    if (getType() == coEditorValueWidget::Info)
    {
        aValuesCheckBox->setStyleSheet("QCheckBox::indicator  background-color: aliceblue; ");
        connect(deleteValueAction, SIGNAL(triggered()), this, SLOT(explainShowInfoButton()));
    }
    else
        connect(deleteValueAction, SIGNAL(triggered()), this, SLOT(suicide()));

    if (value.toLower() == "true" || value.toLower() == "on" || value == "1")
    {
        aValuesCheckBox->setCheckState(Qt::Checked);
    }
    else if (value.toLower() == "false" || value.toLower() == "off" || value == "0")
    {
        aValuesCheckBox->setCheckState(Qt::Unchecked);
    }
    else
        aValuesCheckBox->setCheckState(Qt::PartiallyChecked);

    layout->addWidget(valueLabel);
    layout->addItem(spacerItem);
    layout->addWidget(aValuesCheckBox);

    setLayout(layout);

    connect(aValuesCheckBox, SIGNAL(stateChanged(int)), this, SLOT(save(int)));
}

void coEditorBoolValueWidget::save(int state)
{
    //    if (this->property("infoWidget").toBool() ) hide();
    //    this->setProperty("infoWidget", false); //NOTE not needed, just change type
    if (fType == coEditorValueWidget::Info)
    {
        fType = coEditorValueWidget::Bool;
        disconnect(deleteValueAction, SIGNAL(triggered()), this, SLOT(explainShowInfoButton()));
        connect(deleteValueAction, SIGNAL(triggered()), this, SLOT(suicide()));
    }
    if (state == 0)
        emit saveValue(fvariable, "false");
    else if (state == 2)
        emit saveValue(fvariable, "true");
}

void coEditorBoolValueWidget::suicide()
{
    emit deleteValue(fvariable);
    // make this a infoWidget
    fType = coEditorValueWidget::Info;
    aValuesCheckBox->setCheckState(Qt::PartiallyChecked);
    aValuesCheckBox->setStyleSheet("QCheckBox::indicator  background-color: aliceblue; ");
    connect(deleteValueAction, SIGNAL(triggered()), this, SLOT(explainShowInfoButton()));
}

void coEditorBoolValueWidget::undo()
{
    //aValuesCheckBox->undo();
}
