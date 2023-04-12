/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   10/11/2010
 **
 **************************************************************************/

#include "catalogtreewidget.hpp"

 // Data //
 //
#include "src/data/oscsystem/oscelement.hpp"
#include "src/data/oscsystem/oscbase.hpp"
#include "src/data/projectdata.hpp"
#include "src/data/changemanager.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"
#include "src/gui/tools/osceditortool.hpp"
#include "src/gui/tools/toolmanager.hpp"
#include "src/gui/oscsettings.hpp"

// MainWindow//
//
#include "src/mainwindow.hpp"

//Settings//
//
#include "src/settings/projectsettings.hpp"

// Editor //
//
#include "src/graph/editors/osceditor.hpp"

// Commands //
//
#include "src/data/commands/osccommands.hpp"
#include "src/data/commands/dataelementcommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"

// OpenScenario //
//
#include <OpenScenario/schema/oscObject.h>
#include <OpenScenario/oscObjectBase.h>
#include <OpenScenario/oscMember.h>
#include <OpenScenario/oscMemberValue.h>
#include <OpenScenario/oscCatalog.h>

#include <QWidget>
#include <QDockWidget>
#include <QMouseEvent>
#include <QDrag>
#include <QMimeData>

using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

CatalogTreeWidget::CatalogTreeWidget(MainWindow *mainWindow, OpenScenario::oscCatalog *catalog)
    : QTreeWidget()
    , mainWindow_(mainWindow)
    , oscEditor_(NULL)
    , currentTool_(ODD::TNO_TOOL)
    , oscElement_(NULL)
    , catalog_(catalog)
    , currentMember_(NULL)
{
    init();
}

CatalogTreeWidget::~CatalogTreeWidget()
{
    if (oscElement_)
    {
        oscElement_->detachObserver(this);
    }

    if (base_)
    {
        base_->detachObserver(this);
    }
}

//################//
// FUNCTIONS      //
//################//

void
CatalogTreeWidget::init()
{

    // Connect to DockWidget to receive raise signal//
    //

    projectWidget_ = mainWindow_->getActiveProject();
    projectData_ = projectWidget_->getProjectData();

    // OpenScenario Element base //
    //
    base_ = projectData_->getOSCBase();
    base_->attachObserver(this);
    openScenarioBase_ = catalog_->getBase();
    directoryPath_ = QString::fromStdString(catalog_->Directory->path.getValue());

    // Connect with the ToolManager to send the selected signal or object //
    //
    toolManager_ = mainWindow_->getToolManager();
    if (toolManager_)
    {
        connect(this, SIGNAL(toolAction(ToolAction *)), toolManager_, SLOT(toolActionSlot(ToolAction *)));
    }

    //   setSelectionMode(QAbstractItemView::ExtendedSelection);
    setUniformRowHeights(true);
    setIndentation(6);

    // Signals Widget //
    //
    setColumnCount(3);
    setColumnWidth(0, 180);
    setColumnWidth(1, 30);

    setHeaderHidden(true);

    catalogName_ = catalog_->getCatalogName();
    catalogType_ = "osc" + catalogName_;


    //parse all files only if it has not already been parsed
    //store object name and filename in map
    if (catalog_->getNumObjects() == 0)
    {
        catalog_->fastReadCatalogObjects();
    }

    connect(this, SIGNAL(itemClicked(QTreeWidgetItem *, int)), this, SLOT(onItemClicked(QTreeWidgetItem *, int)));
    setDragEnabled(true);
    createTree();
}

struct TreeDataTyoe {
    QString str;
    OpenScenario::oscObjectBase *obj;
};
Q_DECLARE_METATYPE(TreeDataTyoe);

void
CatalogTreeWidget::createTree()
{
    clear();

    QList<QTreeWidgetItem *> rootList;

    // emtpy item to create new elements //
    //
    QTreeWidgetItem *item = new QTreeWidgetItem();
    item->setText(0, "New Element");
    rootList.append(item);

    // add all catalog members //
    //
    if (catalog_)
    {

        const OpenScenario::oscCatalog::ObjectsMap objects = catalog_->getObjectsMap();
        for (OpenScenario::oscCatalog::ObjectsMap::const_iterator it = objects.begin(); it != objects.end(); it++)
        {
            QString elementName = QString::fromStdString(it->first);
            OpenScenario::oscObjectBase *obj = it->second.object;
            if (obj)
            {
                /*  OpenScenario::oscMember *member = obj->getMember("name");
                        if (member->exists())
                        {
                            oscStringValue *sv = dynamic_cast<oscStringValue *>(member->getOrCreateValue());
                            QString text = QString::fromStdString(sv->getValue());
                            elementName = text + "(" + elementName + ")";
                        } */
                elementName = "Loaded(" + elementName + ")";
            }
            else
            {
                elementName = "NotLoaded(" + elementName + ")";
            }

            QTreeWidgetItem *item = new QTreeWidgetItem();
            /*    TreeDataTyoe td;
                            td.str = elementName;
                            td.obj = obj;
                            item->setData(0, Qt::UserRole, qVariantFromValue<TreeDataTyoe>(td));  */
            item->setText(0, elementName);
            item->setFlags(Qt::ItemIsDragEnabled | Qt::ItemIsSelectable | Qt::ItemIsEnabled);


            rootList.append(item);
        }
    }

    insertTopLevelItems(0, rootList);
}

QTreeWidgetItem *CatalogTreeWidget::getItem(const QString &name)
{
    QTreeWidgetItemIterator it(this);
    while (*it) {
        if ((*it)->text(0).contains(name))
            return (*it);
        ++it;
    }

    return NULL;
}

QTreeWidgetItem *CatalogTreeWidget::getItem(OpenScenario::oscObjectBase *obj)
{
    /* QTreeWidgetItemIterator it(this);
        while (*it)
        {
            if ((*it)->data(0, Qt::UserRole).value<TreeDataTyoe>().obj == obj)
            {
                return (*it);
            }
            ++it;
        } */

    const OpenScenario::oscCatalog::ObjectsMap objects = catalog_->getObjectsMap();
    for (OpenScenario::oscCatalog::ObjectsMap::const_iterator it = objects.begin(); it != objects.end(); it++)
    {
        if (it->second.object == obj)
        {
            return getItem(QString::fromStdString(it->first));
        }
    }

    return NULL;
}

void
CatalogTreeWidget::setOpenScenarioEditor(OpenScenarioEditor *oscEditor)
{
    oscEditor_ = oscEditor;

    if (!oscEditor)
    {
        clearSelection();
    }
}


//################//
// EVENTS         //
//################//
void
CatalogTreeWidget::onItemClicked(QTreeWidgetItem *item, int column)
{
    // if (oscEditor_)
    {
        oscEditor_->enableSplineEditing(false);
        if (item)
        {
            toolManager_->activateOSCObjectSelection(false);

            const QString text = item->text(0);
            currentTool_ = ODD::TOS_ELEMENT;

            if (text == "New Element")
            {
                // Group undo commands
                //
                projectData_->getUndoStack()->beginMacro(QObject::tr("New Catalog Object"));

                if (oscElement_ && oscElement_->isElementSelected())
                {
                    DeselectDataElementCommand *command = new DeselectDataElementCommand(oscElement_, NULL);
                    projectWidget_->getTopviewGraph()->executeCommand(command);
                }

                oscElement_ = new OSCElement(text);

                if (oscElement_)
                {
                    oscElement_->attachObserver(this);


                    // refid vergeben, pruefen ob Datei schon vorhanden, path von vehicleCatalog?, neue Basis fuer catalog?
                    // Element anlegen
                    QString filePath;
                    std::string refId = catalog_->generateRefId();

                    OpenScenario::oscObjectBase *obj = NULL;
                    if (OSCSettings::instance()->loadDefaults())
                    {
                        obj = catalog_->readDefaultXMLObject(filePath.toStdString(), catalogName_, catalogType_);
                    }

                    oscCatalogFile *catalogFile = catalog_->getCatalogFile(catalogName_, directoryPath_.toStdString());
                    if (!obj)
                    {

                        AddOSCObjectCommand *command = new AddOSCObjectCommand(catalog_, base_, catalog_->getCatalogType(), oscElement_, catalogFile->srcFile);
                        if (command->isValid())
                        {
                            projectWidget_->getTopviewGraph()->executeCommand(command);
                        }

                        /*   AddOSCObjectCommand *command = new AddOSCObjectCommand(catalog_, base_, catalogType_, oscElement_, oscSourceFile);
                                    if (command->isValid())
                                    {
                                        projectWidget_->getTopviewGraph()->executeCommand(command);
                                    } */

                        obj = oscElement_->getObject();
                    }

                    AddOSCCatalogObjectCommand *addCatalogObjectCommand = new AddOSCCatalogObjectCommand(catalog_, refId, obj, catalogFile, base_, oscElement_);

                    if (addCatalogObjectCommand->isValid())
                    {
                        projectWidget_->getTopviewGraph()->executeCommand(addCatalogObjectCommand);

                        if (obj)
                        {
                            /*      std::string name = "name";
                                                        SetOSCValuePropertiesCommand<std::string> *setPropertyCommand = new SetOSCValuePropertiesCommand<std::string>(oscElement_, obj, name, text.toStdString());
                                                        projectWidget_->getTopviewGraph()->executeCommand(setPropertyCommand); */

                                                        //       catalog_->writeToDisk();
                                                        //       obj->writeToDisk();
                        }
                    }
                }

                projectData_->getUndoStack()->endMacro();
            }
            else if (text != "")
            {
                // Group undo commands
                //
                projectData_->getUndoStack()->beginMacro(QObject::tr("Load Catalog Object"));

                if (oscElement_ && oscElement_->isElementSelected())
                {
                    DeselectDataElementCommand *command = new DeselectDataElementCommand(oscElement_, NULL);
                    projectWidget_->getTopviewGraph()->executeCommand(command);
                }


                std::string refId = text.toStdString();
                QStringList noBrackets = text.split("(");
                if (noBrackets.size() > 1)
                {
                    refId = noBrackets[1].remove(")").toStdString();
                }

                OpenScenario::oscObjectBase *oscObject = catalog_->getCatalogObject(refId);
                if (oscObject)
                {
                    oscElement_ = base_->getOrCreateOSCElement(oscObject);
                }
                else
                {
                    oscElement_ = new OSCElement(text);

                    if (oscElement_)
                    {
                        oscElement_->attachObserver(this);

                        LoadOSCCatalogObjectCommand *command = new LoadOSCCatalogObjectCommand(catalog_, refId, base_, oscElement_);
                        projectWidget_->getTopviewGraph()->executeCommand(command);
                    }
                }

                if (oscElement_)
                {
                    SelectDataElementCommand *command = new SelectDataElementCommand(oscElement_, NULL);
                    projectWidget_->getTopviewGraph()->executeCommand(command);
                }

                projectData_->getUndoStack()->endMacro();

                // Set a tool //
                //
                OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(currentTool_, text);
                emit toolAction(action);
                delete action;

            }
        }


        //  QTreeWidget::selectionChanged(selected, deselected);
    }
    /* else
        {
            clearSelection();
            clearFocus();
        }*/

}

void CatalogTreeWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
        dragStartPosition_ = event->pos();

    QTreeWidget::mousePressEvent(event);

    onItemClicked(itemAt(event->pos()), 0);
}

void CatalogTreeWidget::mouseMoveEvent(QMouseEvent *event)
{
    QTreeWidget::mouseMoveEvent(event);

    if (!(event->buttons() & Qt::LeftButton))
        return;
    if ((event->pos() - dragStartPosition_).manhattanLength() < QApplication::startDragDistance())
        return;

    //if(oscElement_.is)
    QDrag *drag = new QDrag(this);
    QMimeData *mimeData = new QMimeData;

    OpenScenario::oscMember *nameMember = oscElement_->getObject()->getMember("name");
    OpenScenario::oscStringValue *nameMemberValue = dynamic_cast<oscStringValue *> (nameMember->getValue());
    std::string entryName = nameMemberValue->getValue();

    mimeData->setData("text/plain", QByteArray::fromStdString(entryName));
    drag->setMimeData(mimeData);

    Qt::DropAction dropAction = drag->exec(Qt::CopyAction | Qt::MoveAction);

}
//################//
// SLOTS          //
//################//
void
CatalogTreeWidget::onVisibilityChanged(bool visible)
{
    if (visible && oscEditor_)
    {
        oscEditor_->catalogChanged(catalog_);
    }

    clearSelection();

    if (oscElement_ && oscElement_->isElementSelected())
    {
        DeselectDataElementCommand *command = new DeselectDataElementCommand(oscElement_, NULL);
        projectWidget_->getTopviewGraph()->executeCommand(command);
    }
}


//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
CatalogTreeWidget::updateObserver()
{
    /*    if (isInGarbage())
        {
            return; // will be deleted anyway
        }*/

        /* int changes = base_->getOSCBaseChanges();
            if ((changes & OSCBase::COSC_ElementChange) && !oscElement_->getOSCBase())
            {
                createTree();
                return;
            } */

    if (!oscElement_)
    {
        return;
    }

    // Object name //
    //
    int changes = oscElement_->getOSCElementChanges();

    if (changes & OSCElement::COE_ParameterChange)
    {
        OpenScenario::oscObjectBase *obj = oscElement_->getObject();

        OpenScenario::oscMember *member = obj->getMember("name");
        if (member->exists())
        {
            oscStringValue *sv = dynamic_cast<oscStringValue *>(member->getOrCreateValue());
            QString text = QString::fromStdString(sv->getValue());
            QTreeWidgetItem *currentEditedItem = getItem(oscElement_->getObject());

            if (currentEditedItem != NULL)
            {
                QString elementName = "Loaded(" + text + ")";
                if (currentEditedItem && (elementName != currentEditedItem->text(0)))
                {
                    //OpenScenario::oscObjectBase *oscObject = catalog_->getCatalogObject(currentEditedItem->text(0));
                    catalog_->renameCatalogObject(obj, text.toStdString());
                    currentEditedItem->setText(0, elementName);

                    // Update Editor //
                    //
                    OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(currentTool_, text);
                    emit toolAction(action);
                    delete action;
                }
            }
        }
    }


    changes = oscElement_->getDataElementChanges();
    if ((changes & DataElement::CDE_DataElementAdded) || (changes & DataElement::CDE_DataElementRemoved))
    {
        createTree();
    }
    else if (changes & DataElement::CDE_SelectionChange)
    {
        /*  OpenScenario::oscObjectBase *obj = oscElement_->getObject();

                OpenScenario::oscMember *member = obj->getMember("name");
                if (member->exists())
                {
                    QTreeWidgetItem *currentEditedItem = selectedItems().at(0);

                    currentEditedItem->setSelected(oscElement_->isElementSelected());
                } */
    }

}
