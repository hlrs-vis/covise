/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "coEditorEntryWidget.h"
#include "coEditorGroupWidget.h"
#include "coEditorValidatedQLineEdit.h"
#include <QtGui>
#include <QHBoxLayout>
#include <QAction>
#include <QGroupBox>
#include <QInputDialog>
#include <QSpacerItem>
#include "coEditorValueWidget.h"
#include "coEditorBoolValueWidget.h"
#include "coEditorTextValueWidget.h"

#include <config/coConfigEntryToEditor.h>

#include <iostream> //only for cerr

using namespace covise;

// Konstruktor with coConfigEntry
coEditorEntryWidget::coEditorEntryWidget(QWidget *parent, coEditorGroupWidget *group,
                                         coConfigEntry *entry, const QString &name)
    : QWidget(parent)
{
    //entriesParent = parent;
    groupWidget = group;
    entryWidgetName = name;
    singleEntry = false;

    createConstruct();

    if (entry)
    {
        rootEntry = entry;
        if (entryWidgetName.isEmpty()) // if no name was given in constructor, name will be full path:name.
        {
            entryWidgetName = entry->getPath().section(".", 1, -2) + (".") + (entry->getName());
        }
        // set the name of this object - coEditorGroupWidget::update  will search for it.
        setObjectName(entryWidgetName);
        //NOTE no schemaInfo, no further processing
        info = rootEntry->getSchemaInfos();

        if (info)
        {
            examineEntry();
        }
        //       // means entry is not mentioned in schema, ALL group
        //       else
        //       {
        //          valuesOfEntryGroup->setTitle(rootEntry->getPath().section ('.', 1)) ;
        //          valuesOfEntryGroup->setToolTip("This entry is NOT mentioned in schema");
        //          valuesOfEntryGroup->setDisabled (1);
        // //          valuesOfEntryGroup->hide();
        //       }

        // set section (used by save and delete) - same as entryWidgetName
        // e.g COVER.Plugin.AKToolbar.Shortcut -- LOCAL ... and Shortcut is cut then  Shortcut:Main - is added to have COVER.Plugin.AKToolbar.Shortcut:Main
        section = entry->getPath().section(".", 1, -2) + (".") + (entry->getName());

        connect(deleteAction, SIGNAL(triggered()), this, SLOT(suicide()));
        //       connect(deleteAction , SIGNAL(triggered()), group, SLOT(deleteRequest(QWidget*)));
        connect(moveToHostAction, SIGNAL(triggered()), this, SLOT(moveToHost()));
    }
}

// Konstruktor with coConfigSchemaInfos
coEditorEntryWidget::coEditorEntryWidget(QWidget *parent, coEditorGroupWidget *group,
                                         coConfigSchemaInfos *infos, const QString &name)
    : QWidget(parent)
{
    //    entriesParent = parent;
    //    hasSchema = 1;
    groupWidget = group;
    entryWidgetName = name;
    singleEntry = false;
    rootEntry = 0;

    createConstruct();

    QString boxTip;
    //kind of useless a schemaInfo constructor if schemaInfo is Null, anyway, we check
    if (infos)
    {
        info = infos;
        if (entryWidgetName.isEmpty()) // if no name was given in constructor, name of this widget will be full path + name.
        {
            // cut LOCAL or GLOBAL
            entryWidgetName = info->getElementPath().section("AL.", 1) + (".") + (info->getElement());
        }
        setObjectName(entryWidgetName);

        examineEntry();

        valuesOfEntryGroup->setToolTip(boxTip);
        //mark infoWidget with a lightblue
        QPalette palette = valuesOfEntryGroup->palette();
        palette.setColor(QPalette::Active, QPalette::Window, QColor("lightblue"));
        // palette.setColor(QPalette::Active, QPalette::Base, QPalette::Window);
        valuesOfEntryGroup->setPalette(palette);

        // set section (used by save and delete)
        section = info->getElementPath().section("AL.", 1) + "." + info->getElement();

        connect(deleteAction, SIGNAL(triggered()), this, SLOT(explainShowInfoButton()));
        connect(this, SIGNAL(nameAdded(const QString &, coConfigSchemaInfos *)),
                groupWidget, SLOT(updateInfoWidgets(const QString &, coConfigSchemaInfos *)));
    }
    else
    {
        // explode
    }
}

// setup the layout and the context menu for this coEditorEntryWidget
void coEditorEntryWidget::createConstruct()
{

    /*QGroupBox **/ valuesOfEntryGroup = new QGroupBox;
    valuesOfEntryGroup->setObjectName(QString::fromUtf8("valuesOfEntryGroup"));
    valuesOfEntryGroup->setFlat(0);
    valuesOfEntryGroup->setAttribute(Qt::WA_AlwaysShowToolTips, 1);

    entryWidgetLayout = new QVBoxLayout;
    entryWidgetLayout->setObjectName(QString::fromUtf8("entryWidgetLayout"));
    entryWidgetLayout->setSpacing(1);
    entryWidgetLayout->setMargin(1);
    valuesOfEntryGroup->setLayout(entryWidgetLayout);

    QVBoxLayout *entryWidgetMainLayout = new QVBoxLayout;
    entryWidgetMainLayout->setObjectName(QString::fromUtf8("entryWidgetMainLayout"));
    entryWidgetMainLayout->addWidget(valuesOfEntryGroup);
    entryWidgetMainLayout->insertStretch(1, 2);
    entryWidgetMainLayout->setSpacing(3);
    entryWidgetMainLayout->setMargin(2);
    setLayout(entryWidgetMainLayout);

    deleteAction = new QAction(tr("Delete this entry, with all its values ?"), this);
    deleteAction->setStatusTip(tr("Delete this entry for the active host."));

    moveToHostAction = new QAction(tr("Move this entry to another host ?"), this);
    moveToHostAction->setStatusTip(tr("Move"));

    addAction(deleteAction);
    // addAction(moveToHostAction); //doesnt work in coConfig libary

    setContextMenuPolicy(Qt::ActionsContextMenu);
}

// get data from entry, call then fitting function to display
void coEditorEntryWidget::examineEntry()
{
    QRegExp rx;
    QString value, attrDescription, attrReadableRule, attrDefaultValue;
    bool infoWidget = 0;

    // set title and description
    valuesOfEntryGroup->setToolTip(info->getElementDescription());
    // fetch allowed attributes from schemaInfos
    QList<QString> attributes = info->getAttributes();
    if (attributes.isEmpty())
    {
        // warn no attributes found for this element
        this->hide();
        return;
    }
    // check if element has only one attribute, because then we replace "value" with the name of the entry itself. Also there is no BoxTitle
    if (attributes.count() == 1 && attributes.first() == "value")
    {
        singleEntry = true;
        valuesOfEntryGroup->setTitle(QString());
    }
    else
        valuesOfEntryGroup->setTitle(info->getElementName());

    // iterate over allowed attributes and create widgets for the attributes
    for (QList<QString>::const_iterator item = attributes.begin(); item != attributes.end(); ++item)
    {

        // get value for this attribute from coConfigEntry
        if (rootEntry)
            value = rootEntry->getValue((*item), QString::null);
        //check wether there is a attrData structure for this attribute and get Data
        attrData *attributeData = info->getAttributeData((*item));
        if (attributeData)
        {
            rx.setPattern(attributeData->regularExpressionString);
            attrDescription = attributeData->attrDescription;
            attrReadableRule = attributeData->readableRule;
            attrDefaultValue = attributeData->defaultValue;
            //bool required = attributeData->required;
        }
        //check if it is a real entry or a info entry
        if (value.isEmpty())
        {
            value = attrDefaultValue;
            infoWidget = 1;
        }

        if (!valuesList.contains(*item)) // check if this widget already has a children coEditorValueWidget for this attribute
        {
            valuesList.append(*item);
            // create a new coEditorValueWidget for this attribute depending on its type
            //TODO better way to decide between attribute  boolean or lineEdit
            if (value.toLower() == "true" || value.toLower() == "on" || value.toLower() == "false" || value.toLower() == "off" || attrReadableRule.contains("bool"))
            {
                // if(!attrReadableRule.isEmpty() && attrReadableRule.contains("bool"))
                if (singleEntry) // replace "value" with the name of the entry itself.
                    createBooleanValue(info->getElementName(), value, attrDescription, infoWidget);
                else
                    createBooleanValue((*item), value, attrDescription, infoWidget);
            }
            else
            {
                if (rx.isEmpty() || !rx.isValid())
                {
                    //warn no RegExp from schema
                    rx.setPattern("^.*");
                }
                if (singleEntry) // replace "value" with the name of the entry itself.
                    createQregXpValue(info->getElementName(), value, rx, attrReadableRule, attrDescription, infoWidget);
                else
                    createQregXpValue((*item), value, rx, attrReadableRule, attrDescription, infoWidget);
            }
        }
        else // there is a coEditorValueWidget for this attribute
        {
            // TODO check if value is correct, otherwise (mb entry was changed from outside editor) set. (!infinite loop)
        }
        // reset placeholder
        attrDescription = "";
        attrReadableRule = "";
        attrDefaultValue = "";
        value = "";
        rx.setPattern("^.*");
        infoWidget = 0;
    }
}

void coEditorEntryWidget::createQregXpValue(const QString &valueName, const QString &value,
                                            const QRegExp &rx, const QString &readableAttrRule,
                                            const QString &attributeDescription,
                                            bool empty, bool required)
{
    if (valueName.isEmpty() /*|| value .isEmpty()*/)
        return;

    QString textWidgetName = valueName;
    coEditorTextValueWidget *textWidget;
    if (empty) // means this will be a infoWidget
    {
        if (textWidgetName.compare("name") == 0)
        {
            textWidget = new coEditorTextValueWidget(0, textWidgetName, coEditorValueWidget::InfoName);
            connect(textWidget, SIGNAL(nameAdded(const QString &)),
                    this, SLOT(nameAddedSlot(const QString &)));
        }
        else
        {
            textWidget = new coEditorTextValueWidget(0, textWidgetName, coEditorValueWidget::Info);
        }
    }
    else
        textWidget = new coEditorTextValueWidget(0, textWidgetName, coEditorValueWidget::Text);

    textWidget->setValue(valueName, value, readableAttrRule, attributeDescription, required, rx);
    entryWidgetLayout->addWidget(textWidget);

    connect(textWidget, SIGNAL(saveValue(const QString &, const QString &)),
            this, SLOT(saveValue(const QString &, const QString &)));

    connect(textWidget, SIGNAL(deleteValue(const QString &)),
            this, SLOT(deleteValue(const QString &)));
}

void coEditorEntryWidget::createBooleanValue(const QString &valueName, const QString &value,
                                             const QString &attributeDescription,
                                             bool empty, bool required)
{
    QString boolWidgetName = valueName;
    coEditorBoolValueWidget *boolWidget;
    if (empty)
        boolWidget = new coEditorBoolValueWidget(0, boolWidgetName, coEditorValueWidget::Info);
    else
        boolWidget = new coEditorBoolValueWidget(0, boolWidgetName, coEditorValueWidget::Bool);
    boolWidget->setValue(valueName, value, 0, attributeDescription, required);
    entryWidgetLayout->addWidget(boolWidget);

    boolWidget->setProperty("infoWidget", empty);

    connect(boolWidget, SIGNAL(saveValue(const QString &, const QString &)),
            this, SLOT(saveValue(const QString &, const QString &)));

    connect(boolWidget, SIGNAL(deleteValue(const QString &)),
            this, SLOT(deleteValue(const QString &)));
}

void coEditorEntryWidget::saveValue(const QString &variable, const QString &value)
{

    QString trueVariable = variable;
    QString sectionTmp = section; // setName() may change section to new name. but to save right the old name is needed.
    if (singleEntry) // when this entry only has one attribute "value", it was replaced by the entries name - here we switch it back
    {
        trueVariable = "value";
    }

    // check if the variable name has been changed, and is free.
    if (trueVariable.compare("name") == 0)
    {
        if (!setName(value))
            return; // if name is impossible dont save.
    }

    // if we have an coConfigEntry for this widget, (we may save directly). Otherwise we emit the saveValue signal
    if (rootEntry)
    {
        //       cerr << "coEditorMainWindow::set var "  << trueVariable.toLatin1().data() << " value " << value.toLatin1().data() << endl;
        //       rootEntry->setValue(trueVariable, value, QString());  // save direct to entry
        //       section = rootEntry->getPath().section(".",0, -2);  // e.g COVER.Plugin.AKToolbar.Shortcut -- Shortcut is cut
        //       section.append(".");
        //       section.append(rootEntry->getName());  // e.g Shortcut:Main - is added to have COVER.Plugin.AKToolbar.Shortcut:Main
    }
    else
    {
        // Change infoWidget behaviour to normal coEditorEntryWidget behave. when this is a virgin infoWidget and this is its first time being used
        if (this->property("infoWidget").toBool()) //(info)
        {
            tranformToRealEntry();
            groupWidget->setOutOfDate(); // tell parent coEditorGroupWidget that it is outOfDate
        }
    }

    // send save signal
    emit saveValue(trueVariable, value, sectionTmp);
}

void coEditorEntryWidget::deleteValue(const QString &variable)
{

    QString trueVariable = variable;
    if (singleEntry) // when this entry only has one attribute "value", it was replaced by the entries name - here we switch it back
    {
        trueVariable = "value";
    }

    // Remove that item from the valuesList
    int index = valuesList.indexOf(trueVariable);
    if (index != -1)
        valuesList.removeAt(index);

    emit deleteValue(trueVariable, section);

    if (valuesList.isEmpty()) // if all values have been deleted remove this widget
    {
        emit deleteRequest(this);
    }

    if (trueVariable.compare("name") == 0) // if name was deleted, delete all values
    {
        suicide();
    }
}

// only called by widgets with real values: initiated by contextMenu-delete
// delete all values directly, then initiate destruction of this window
void coEditorEntryWidget::suicide()
{
    if (rootEntry)
    {
        QList<QString> attributes = /*rootEntry->getSchemaInfos()*/ info->getAttributes(); // fetch allowed attributes from schemaInfos
        if (!attributes.isEmpty())
        {
            for (QList<QString>::const_iterator item = attributes.begin(); item != attributes.end(); ++item)
            {
                rootEntry->deleteValue(*item, QString::null);
            }
        }
    }
    else
    {
        // get all checkboxes and lineEdits , then delete these values
        QList<coEditorValueWidget *> widgetList = valuesOfEntryGroup->findChildren<coEditorValueWidget *>();
        while (!widgetList.isEmpty())
        {
            coEditorValueWidget *wid = widgetList.takeFirst();
            if (!(wid->getType() == coEditorValueWidget::InfoName || wid->getType() == coEditorValueWidget::Info))
            {
                QString variable = wid->objectName();
                emit deleteValue(variable, section);
            }
        }
    }

    emit deleteRequest(this);
}

//get targetHost, collect all children, emit save with targethost
void coEditorEntryWidget::moveToHost()
{
    bool ok;
    QString host = QInputDialog::getText(this, tr("Move to Host"),
                                         tr("Type name of host to move: "), QLineEdit::Normal,
                                         0, &ok);
    if (ok && !host.isEmpty())
    {
        // collect all coEditorValueWidget and save all that arent only infoWidgets
        QList<coEditorValueWidget *> coEditorValueWidgets = findChildren<coEditorValueWidget *>();
        if (!coEditorValueWidgets.isEmpty())
        {
            for (QList<coEditorValueWidget *>::const_iterator iter = coEditorValueWidgets.begin(); iter != coEditorValueWidgets.end(); ++iter)

            {
                if ((*iter)->getType() != coEditorValueWidget::Info || (*iter)->getType() != coEditorValueWidget::InfoName)
                {
                    //dies in coConfigGroup
                    emit saveValue((*iter)->getVariable(), (*iter)->getValue(), section, host);
                }
            }
        }
        // Because its "moveToHost" and not "copyToHost" -> entry kills itself.  PS. Have to call outOfDate for the others host GroupWidget?!
        suicide();
    }
}

// tranform a coEditorEntryWidget that was created from a SchemaInfo without content (a so called infoWidget) to a real one
void coEditorEntryWidget::tranformToRealEntry()
{
    setProperty("infoWidget", false);
    // stop reaction to showInfoWidgets button
    disconnect(groupWidget, SIGNAL(hideYourselfInfoWidgets()), this, SLOT(hide()));
    disconnect(groupWidget, SIGNAL(showYourselfInfoWidgets()), this, SLOT(show()));

    // change behaviour for delete context menu
    disconnect(deleteAction, SIGNAL(triggered()), this, SLOT(explainShowInfoButton()));
    connect(deleteAction, SIGNAL(triggered()), this, SLOT(suicide()));

    groupWidget->removeFromInfoWidgetList(info);
}

// called by coEditorTextValueWidget when a nameInfoWidget saves a new name
void coEditorEntryWidget::nameAddedSlot(const QString &value)
{
    QString sectionTmp = section; // setName() may change section to new name. but to save right the old name is needed.
    // check if name is possible, set it then, otherwise inform user. returns true if possible.
    if (setName(value))
    {
        if (this->property("infoWidget").toBool())
            tranformToRealEntry();

        // save Value
        emit saveValue("name", value, sectionTmp);

        // tell coEditorGroupWidget that this is no longer a infoWidget. GroupWidget then may create a new one
        emit nameAdded(value, info);
    }
}

//called by groupWidget, when an entry was changed (Observer).
void coEditorEntryWidget::refresh(coConfigEntry *subject)
{
    if (rootEntry == 0)
    {
        rootEntry = subject;
    }
    QString newName = rootEntry->getName().section(":", 1);
    //setName(newName);
}

void coEditorEntryWidget::explainShowInfoButton()
{
    QMessageBox::information(this, tr("Covise Config Editor"),
                             tr("There is no need to delete a empty Field.\n"
                                "To hide this field, press the Button with the \n"
                                "bkue I at the top. \n"));
}

coConfigEntry *coEditorEntryWidget::getEntry()
{
    return rootEntry;
}

coConfigSchemaInfos *coEditorEntryWidget::getSchemaInfo()
{
    return info;
}

// check if name is possible, set it then. Otherwise inform user and undo change.
bool coEditorEntryWidget::setName(const QString &name)
{
    if (groupWidget->nameIsFree(name, this))
    {
        // change objectName of this coEditorEntryWidget to new name
        entryWidgetName = entryWidgetName.section(":", 0, 0);
        QString nameValue = ":" + name;
        entryWidgetName.append(nameValue);
        setObjectName(entryWidgetName);
        section = section.section(":", 0, 0) + nameValue;
        return true;
    }
    else
    {
        QMessageBox::information(this, tr("Covise Config Editor - EntryWidget"),
                                 tr("This name is already ocupied. Choose another one!\n"));
        // set Value back to its original state
        coEditorValueWidget *searchedEntryWidget = findChild<coEditorValueWidget *>("name");
        if (searchedEntryWidget != 0)
        {
            searchedEntryWidget->undo();
        }
        return false;
    }
}
