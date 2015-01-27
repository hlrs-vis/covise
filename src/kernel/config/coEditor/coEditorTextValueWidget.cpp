/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coEditorTextValueWidget.h"
#include "coEditorValidatedQLineEdit.h"
#include <QHBoxLayout>
#include <QLabel>
#include <QAction>
#include <QRegExpValidator>

coEditorTextValueWidget::coEditorTextValueWidget(QWidget *parent, const QString name, Type type)
    : coEditorValueWidget(parent, type)
{
    widgetName = name;
    defaultValue = "";
    //    if (getType() == coEditorValueWidget::INFO)
    //    {
    //      setStyleSheet ("background-color: aliceblue");
    //    }
    //    else setStyleSheet ("background-color:");
}

coEditorTextValueWidget::~coEditorTextValueWidget()
{
}

void coEditorTextValueWidget::setValue(const QString &valueName, const QString &value,
                                       const QString &readableAttrRule,
                                       const QString &attributeDescription,
                                       bool, const QRegExp &rx)
{
    fvariable = valueName;
    fvalue = value;

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

    valueLineEdit = new coEditorValidatedQLineEdit(this);
    valueLineEdit->setObjectName(valueName);
    valueLineEdit->setStatusTip(readableAttrRule);
    QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(3), static_cast<QSizePolicy::Policy>(0));
    sizePolicy.setHorizontalStretch(2);
    sizePolicy.setVerticalStretch(0);
    sizePolicy.setHeightForWidth(valueLineEdit->sizePolicy().hasHeightForWidth());
    valueLineEdit->setSizePolicy(sizePolicy);
    valueLineEdit->setMaximumSize(QSize(16777215, 22));
    valueLineEdit->setBaseSize(QSize(200, 22));
    valueLineEdit->setAlignment(Qt::AlignRight);
    valueLineEdit->setDragEnabled(true);

    if (!rx.isEmpty())
    {
        QValidator *validator = new QRegExpValidator(rx, this);
        valueLineEdit->setValidator(validator);
    }
    // returnPressed is only send, if expression is valid. then we need to call coConfigEntry->setValue
    // connect(valueLineEdit, SIGNAL(returnPressed()), this, SLOT(commitNewEntryValueData()));
    valueLineEdit->setText(value); //setText does not call validate
    valueLineEdit->setModified(false);

    //create Contextmenu - Delete value
    deleteValueAction = new QAction(tr("Delete this Value ?"), valueLabel);
    deleteValueAction->setStatusTip(tr("Delete this value ?"));
    addAction(deleteValueAction);
    setContextMenuPolicy(Qt::ActionsContextMenu);

    if (getType() == coEditorValueWidget::Info || getType() == coEditorValueWidget::InfoName
                                                  /*value.isEmpty() &&*/ /*&& !required*/)
    {
        defaultValue = value;
        //
        valueLineEdit->setStyleSheet("background-color: aliceblue");
        connect(deleteValueAction, SIGNAL(triggered()), this, SLOT(explainShowInfoButton()));
    }
    else
    {
        //is valid checks, and sets background color to red if check fails
        /* if (!empty)*/
        valueLineEdit->isValid();
        connect(deleteValueAction, SIGNAL(triggered()), this, SLOT(suicide()));
    }

    valueLineEdit->adjustSize();

    QSize size(250, 45);
    size = size.expandedTo(minimumSizeHint());
    resize(size);

    //connect to save
    connect(valueLineEdit, SIGNAL(editingFinished()), this, SLOT(save()));
    //    connect ( valueLineEdit, SIGNAL (returnPressed() ), this, SLOT (save() ) );
    //NOTE if we want a nasty nagscreen, connect it here
    //connect ( valueLineEdit, SIGNAL (notValid() ),  , SLOT ( ) );
    connect(valueLineEdit, SIGNAL(notValid()), this, SIGNAL(notValid()));

    layout->addWidget(valueLabel);
    layout->addItem(spacerItem);
    layout->addWidget(valueLineEdit);

    setLayout(layout);
}

void coEditorTextValueWidget::save()
{
    if (valueLineEdit->isModified())
    {
        QString value = valueLineEdit->text();
        //    if (this->property("infoWidget").toBool() ) hide();
        //this->setProperty("infoWidget", false);  fType should be enough, except i do systemwide colorize by property
        if (fType == coEditorValueWidget::InfoName) // this was a infoWidget that now for first time got a name
        {
            emit nameAdded(value); // saveValue is called in coEditorEntryWidget::nameAddedSlot
        }
        else // standard behaviour - call save
        {
            emit saveValue(fvariable, value);
        }

        // change behaviour for "delete this value" context menu
        if (fType == coEditorValueWidget::InfoName || fType == coEditorValueWidget::Info)
        {
            fType = coEditorValueWidget::Text;
            disconnect(deleteValueAction, SIGNAL(triggered()), this, SLOT(explainShowInfoButton()));
            connect(deleteValueAction, SIGNAL(triggered()), this, SLOT(suicide()));
        }

        valueLineEdit->setModified(false);
    }
}

// void coEditorTextValueWidget::move()
// {
//    QString value = valueLineEdit->text();
//
//
// }

void coEditorTextValueWidget::suicide()
{
    // make this a infoWidget
    fType = coEditorValueWidget::Info;
    valueLineEdit->setText(defaultValue);
    valueLineEdit->setModified(false);
    valueLineEdit->setStyleSheet("background-color: aliceblue");
    connect(deleteValueAction, SIGNAL(triggered()), this, SLOT(explainShowInfoButton()));
    //    hide();
    //    delete this;

    emit deleteValue(fvariable);
}

void coEditorTextValueWidget::undo()
{
    valueLineEdit->undo();
}
