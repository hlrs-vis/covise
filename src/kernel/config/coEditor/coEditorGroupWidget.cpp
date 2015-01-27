/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <config/coConfigRoot.h>
#include "coEditorGroupWidget.h"

#include <QtGui>
#include "coEditorEntryWidget.h"

#include <QListWidget>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QScrollArea>

#include "coEditorValidatedQLineEdit.h"

#include <config/kernel/coConfigSchema.h>
#include <config/coConfigSchemaInfosList.h>

#include <iostream> // for DEBUG only

using namespace covise;

//the observer, now this class is the observer itself
//  #include "config/coConfigEditorEntry.h"

coEditorGroupWidget::coEditorGroupWidget(QWidget *parent, const QString &name,
                                         coConfigEntry *entry, coConfigSchemaInfosList *infoList)
{
    mainWin = parent;
    setObjectName(name);
    groupName = name.section(":", 1); // gruppenobjectName = Host:GroupName

    groupList = infoList;
    entries.clear();

    //GroupBox with Name of group eg OpenSG
    QGroupBox *groupGroup = new QGroupBox(name, this);
    groupGroup->setObjectName(QString::fromUtf8("groupGroup"));
    groupGroup->setMinimumSize(100, 100);
    groupGroup->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    //MainLayout of this group widget, still place to add sth outside the SectionBox
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(-1);
    mainLayout->setMargin(2); //platz um die groupBox
    mainLayout->setObjectName(QString::fromUtf8("mainLayout"));

    //Layout of GroupBox, still sth like a helpLine can be added inside the SectionBox
    QVBoxLayout *vboxLayout1 = new QVBoxLayout(groupGroup);
    vboxLayout1->setSpacing(2); //abstand zw items innerhalb der box
    vboxLayout1->setMargin(3); //abstand zw boxRand und items
    vboxLayout1->setObjectName(QString::fromUtf8("vboxLayout1"));
    groupGroup->setLayout(vboxLayout1);

    // Scrollarea inside the GroupBox
    QScrollArea *scrollArea = new QScrollArea(groupGroup);
    scrollArea->setWidgetResizable(1);
    scrollArea->setAlignment(Qt::AlignTop);
    scrollArea->setBackgroundRole(QPalette::Window);
    scrollArea->setFrameShape(QFrame::NoFrame);

    // entries widget are inside here, this is their parentwidget
    groupEntryWidget = new QWidget();
    groupEntryWidget->setObjectName(QString::fromUtf8("groupEntryWidget"));
    groupEntryWidget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    scrollArea->setWidget(groupEntryWidget);
    //Layout for entries, entries widget are added here
    groupGroupLayout = new QVBoxLayout(groupEntryWidget);
    groupGroupLayout->addStretch(1); // add a stretch item that eats empty space
    groupGroupLayout->setMargin(5);
    groupGroupLayout->setObjectName(QString::fromUtf8("groupGroupLayout"));
    groupEntryWidget->setLayout(groupGroupLayout);

    vboxLayout1->addWidget(scrollArea);
    mainLayout->addWidget(groupGroup);
    setLayout(mainLayout);

    if (entry)
    {
        rootEntry = entry;
        fOutOfDate = false;
    }
    if (groupList)
    {
        emit showStatusBar("Has Information from Schema");
    }
    addedInfoWidgets = 0;
}

//observer part
void coEditorGroupWidget::update(coConfigEntry *subject)
{
    if (subject->hasValues())
    {
        QString name = subject->getName();
        QString lookup = subject->getPath().section(".", 1, -2) + "." + subject->getName();
        coEditorEntryWidget *searchedEntryWidget = findChild<coEditorEntryWidget *>(lookup);
        if (searchedEntryWidget != 0)
        {
            searchedEntryWidget->refresh(subject);
        }
    }
    else
        emit showStatusBar("entry has no values");
}

// create new widget for the coConfigEntry. If overwrite is true, the existing widget is deleted beforehand.
void coEditorGroupWidget::addEntry(const QString name, coConfigEntry *entry, bool overwrite)
{
    // before the stretch element
    int positionInLayout = groupGroupLayout->count() - 1;
    //check if element is already declared and has to be changed
    if (entries.contains(name) && overwrite)
    {
        entries.take(name);
        coEditorEntryWidget *declaredWidget = this->findChild<coEditorEntryWidget *>(entry->getPath().section(".", 1, -2) + "." + entry->getName());
        positionInLayout = groupGroupLayout->indexOf(declaredWidget);
        delete declaredWidget;
    }
    else if (overwrite) //delete old infowidget. to create a new one  that now has a entry
    {
        coEditorEntryWidget *declaredWidget = this->findChild<coEditorEntryWidget *>(entry->getPath().section(".", 1, -2) + "." + entry->getName());
        positionInLayout = groupGroupLayout->indexOf(declaredWidget);
        delete declaredWidget;
    }

    if (!entries.contains(name))
    {
        entries.insert(name, entry);
        // attach itself as Observer to all added entries
        entry->attach(*this);
        //    create new EntryWidget for this entry
        coEditorEntryWidget *myEntry = new coEditorEntryWidget(groupEntryWidget, this, entry, "" /*entry->getPath()*/);
        myEntry->setProperty("infoWidget", false);
        connect(myEntry, SIGNAL(saveValue(const QString &, const QString &, const QString &, const QString &)),
                this, SLOT(saveValueEntry(const QString &, const QString &, const QString &, const QString &)));
        connect(myEntry, SIGNAL(deleteValue(const QString &, const QString &)),
                this, SLOT(deleteValueEntry(const QString &, const QString &)));
        connect(myEntry, SIGNAL(deleteRequest(coEditorEntryWidget *)), this, SLOT(deleteRequest(coEditorEntryWidget *)));
        // insert widget before the stretch element
        groupGroupLayout->insertWidget(positionInLayout, myEntry);
        //fOutOfDate = 0;
    }
}

// iterate over given Hash and call addEntry for each.
void coEditorGroupWidget::addEntries(QHash<QString, coConfigEntry *> entriesList, bool overwrite)
{
    QList<QString> entryName = entriesList.keys();
    for (QList<QString>::const_iterator iter = entryName.begin(); iter != entryName.end(); ++iter)
    {
        addEntry((*iter), entriesList.value(*iter), overwrite);
    }
    fOutOfDate = false;
    createInfoWidgets();
}

/// creates all possible entries (infoWidgets) that are not already created from coConfigEntries.
void coEditorGroupWidget::createInfoWidgets()
{
    // only do work once.
    if (!addedInfoWidgets)
    {
        if (!groupList)
            groupList = coConfigSchema::getInstance()->getSchemaInfosForGroup(groupName);
        // is still empty at start
        if (groupList)
        {
            for (QList<coConfigSchemaInfos *>::const_iterator iter = groupList->begin(); iter != groupList->end(); ++iter)
            {
                QString name = (*iter)->getElement(); //this is elementName e.g. VRViewPointPlugin
                //NOTE here add check compositor type, then decide wether to add or not
                // check if it has attributes
                if (!(*iter)->getAttributes().isEmpty())
                {
                    // check if already exists.
                    if (!entries.contains(name) && !infoWidgetList.contains(name))
                    {
                        // create infos coEditorEntryWidget.
                        createInfoWidget(*iter);
                    }
                }
            }
        }
        //infoWidgetList.clear();
        emit hideYourselfInfoWidgets();
        addedInfoWidgets = 1;
    }
}

void coEditorGroupWidget::showInfoWidget(bool show)
{
    // Show or hide empty infoWidgets
    if (show)
    {
        //infoLine->setText("Now all possible Items are displayed");
        emit showYourselfInfoWidgets();
    }
    else
    {
        //infoLine->setText("Only declared Items are displayed");
        emit hideYourselfInfoWidgets();
    }
}

void coEditorGroupWidget::saveValueEntry(const QString &variable, const QString &value, const QString &section, const QString &host)
{
    // add host, then transfer to MainWindow to save the value
    // add host (take current one if not a new one is given)
    QString targetHost = host;
    if (targetHost.isEmpty())
        targetHost = objectName().section(":", 0, 0);

    emit saveValue(variable, value, section, targetHost);
}

void coEditorGroupWidget::deleteValueEntry(const QString &variable, const QString &section)
{
    QString host = objectName().section(":", 0, 0);
    emit deleteValue(variable, section, host);
}

void coEditorGroupWidget::deleteRequest(coEditorEntryWidget *widget)
{
    QString name = widget->objectName();
    // create new InfoWidget if neccessary
    if (!infoWidgetList.contains(widget->getSchemaInfo()->getElement()))
    {
        createInfoWidget(widget->getSchemaInfo());
    }

    widget->deleteLater(); // delete the widget
}

// create new InfoWidget because for the old one "name" has been set. //called by coEditorEntryWidget::nameadded
void coEditorGroupWidget::updateInfoWidgets(const QString &newName, coConfigSchemaInfos *info)
{
    //    QString path = info->getElementPath();
    //    QString cleanName = newName;

    QString fullName = info->getElementName() + ":" + newName;
    entries.insert(fullName, NULL);
    // create new InfoWidget if neccessary
    if (!infoWidgetList.contains(info->getElement()))
    {
        createInfoWidget(info);
    }
}

void coEditorGroupWidget::removeFromInfoWidgetList(coConfigSchemaInfos *info)
{
    infoWidgetList.removeAt(infoWidgetList.indexOf(info->getElement()));
}

//check if this name is free or ocupied
bool coEditorGroupWidget::nameIsFree(const QString &newName, coEditorEntryWidget *entryWid)
{

    QString fullName = entryWid->getSchemaInfo()->getElementName() + ":" + newName;
    if (entries.keys().contains(fullName)) // TODO Check if neccessary - see XsModel.compositorType
    {
        return false;
    }
    else
    {
        return true;
    }
}

/// create new infoWidget and add it to the layout
void coEditorGroupWidget::createInfoWidget(coConfigSchemaInfos *info)
{
    coEditorEntryWidget *myEntry = new coEditorEntryWidget(groupEntryWidget, this, info, "" /* info->getElementPath()*/);
    myEntry->setProperty("infoWidget", true);
    connect(myEntry, SIGNAL(saveValue(const QString &, const QString &, const QString &, const QString &)),
            this, SLOT(saveValueEntry(const QString &, const QString &, const QString &, const QString &)));
    connect(myEntry, SIGNAL(deleteValue(const QString &, const QString &)),
            this, SLOT(deleteValueEntry(const QString &, const QString &)));
    connect(myEntry, SIGNAL(deleteRequest(coEditorEntryWidget *)),
            this, SLOT(deleteRequest(coEditorEntryWidget *)));
    // connect (myEntry, SIGNAL( deleteValue(const QString &,const QString &)),
    //this, SLOT(deleteValueEntry(const QString &,const QString &)));
    connect(this, SIGNAL(hideYourselfInfoWidgets()), myEntry, SLOT(hide()));
    connect(this, SIGNAL(showYourselfInfoWidgets()), myEntry, SLOT(show()));
    // insert widget before the stretch element
    groupGroupLayout->insertWidget(groupGroupLayout->count() - 1, myEntry);
    infoWidgetList.append(info->getElement()); // add to infoWidgetList that holds names of all infowidgets.
}

// coEditorMainWindow::workgroup will check this value and collect the sub entries new if outOfDate is true
bool coEditorGroupWidget::outOfDate()
{
    return fOutOfDate;
}

// called from a coEditorEntryWidget with argument when a infoWidget changes to a real coEditorValueWidget. And by coEditorMainWindow::informcoEditorGroupWidgets.
void coEditorGroupWidget::setOutOfDate()
{
    fOutOfDate = true;
}
