/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.03.2010
**
**************************************************************************/

#include "graphelement.hpp"

// Data //
//
#include "src/data/dataelement.hpp"
#include "src/data/commands/dataelementcommands.hpp"

// Graph //
//
#include "src/graph/profilegraph.hpp"
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"

//################//
// CONSTRUCTOR    //
//################//

GraphElement::GraphElement(GraphElement *parentGraphElement, DataElement *dataElement)
    : QObject()
    , QGraphicsPathItem(parentGraphElement)
    , Observer()
    , parentGraphElement_(parentGraphElement)
    , dataElement_(dataElement)
    , hovered_(false)
    , useHighlighting_(true)
    , highlightOpacity_(1.0)
    , normalOpacity_(0.5)
    , isInGarbage_(false)
{
    init();
}

GraphElement::~GraphElement()
{
    // ContextMenu //
    //
    delete contextMenu_;

    // Observer //
    //
    if (dataElement_)
    {
        dataElement_->detachObserver(this);
    }
}

void
GraphElement::init()
{
    updateHighlightingState();

    // ContextMenu //
    //
    contextMenu_ = new QMenu();

    hideMenu_ = new QMenu(tr("Hide"));
    contextMenu_->addMenu(hideMenu_);

    removeMenu_ = new QMenu(tr("Delete"));
    contextMenu_->addMenu(removeMenu_);

    getContextMenu()->addSeparator();

    // Selection //
    //
    setFlag(QGraphicsItem::ItemIsSelectable, false); // not selectable by default

    if (dataElement_)
    {
        // Selection/Hiding //
        //
        setVisible(!dataElement_->isElementHidden());
        setCacheMode(QGraphicsItem::DeviceCoordinateCache);
        //setSelected(dataElement_->isElementSelected()); // will be ignored, if not selectable

        // Observer //
        //
        dataElement_->attachObserver(this);
    }
}

void
GraphElement::setSelectable()
{
    if (dataElement_)
    {
        setFlag(QGraphicsItem::ItemIsSelectable, true);
        setSelected(dataElement_->isElementSelected());
    }
}

void
GraphElement::registerForDeletion()
{
    if (getTopviewGraph())
    {
        getTopviewGraph()->addToGarbage(this);
    }
    else if (getProfileGraph())
    {
        getProfileGraph()->addToGarbage(this);
    }

    notifyDeletion();
}

void
GraphElement::notifyDeletion()
{
    isInGarbage_ = true;
}

ProjectData *
GraphElement::getProjectData() const
{
    if (dataElement_)
    {
        return dataElement_->getProjectData();
    }
    else
    {
        return NULL;
    }
}

ProjectGraph *
GraphElement::getProjectGraph() const
{
    if (getTopviewGraph())
    {
        return getTopviewGraph();
    }
    else if (getProfileGraph())
    {
        return getProfileGraph();
    }
    else
    {
        return NULL;
    }
}

//##################//
// TopviewGraph      //
//##################//

/*! \brief Returns the TopviewGraph this DataElement belongs to.
*
* Returns the TopviewGraph of its parent element. Only root nodes
* like the RoadSystemItem actually save and return it directly.
*/
TopviewGraph *
GraphElement::getTopviewGraph() const
{
    if (parentGraphElement_)
    {
        return parentGraphElement_->getTopviewGraph();
    }
    else
    {
        return NULL;
    }
}

ProfileGraph *
GraphElement::getProfileGraph() const
{
    if (parentGraphElement_)
    {
        return parentGraphElement_->getProfileGraph();
    }
    else
    {
        return NULL;
    }
}

//################//
// GRAPHELEMENT   //
//################//

void
GraphElement::setHovered(bool hovered)
{
    if (hovered_ != hovered)
    {
        hovered_ = hovered;
        updateHighlightingState();
    }
    if (parentGraphElement_)
    {
        parentGraphElement_->setHovered(hovered);
    }
}

void
GraphElement::enableHighlighting(bool enable)
{
    useHighlighting_ = enable;
}

void
GraphElement::setHighlighting(bool highlight)
{
    if (!useHighlighting_)
    {
        return;
    }

    if (highlight)
    {
        setOpacity(highlightOpacity_);
    }
    else
    {
        setOpacity(normalOpacity_);
    }
}

void
GraphElement::setOpacitySettings(double highlightOpacity, double normalOpacity)
{
    highlightOpacity_ = highlightOpacity;
    normalOpacity_ = normalOpacity;
    updateHighlightingState();
}

void
GraphElement::updateHighlightingState()
{
    setHighlighting(hovered_ || (dataElement_ && (dataElement_->isElementSelected() || dataElement_->isChildElementSelected())));
}

//################//
// OBSERVER       //
//################//

void
GraphElement::updateObserver()
{
    // Check //
    //
    //if(!dataElement_)
    //{
    //	return; // unnecessary: no dataElement => no observing
    //}

    // Get change flags //
    //
    int changes = dataElement_->getDataElementChanges();

    // Deletion //
    //
    if ((changes & DataElement::CDE_DataElementDeleted)
        || (changes & DataElement::CDE_DataElementRemoved))
    {
        registerForDeletion();
        return;
    }

    // Hiding //
    //
    if ((changes & DataElement::CDE_HidingChange))
    {
        setVisible(!dataElement_->isElementHidden()); // Do this before handling the selection!
    }

    // Selection //
    //
    if (changes & DataElement::CDE_SelectionChange)
    {
        // Selection //
        //
        if (isSelected() != dataElement_->isElementSelected())
        {
            // DO NOT LET THE OBSERVER CALL itemChange() WHEN YOU ARE ALREADY IN IT!
            setSelected(dataElement_->isElementSelected());
        }

        // Highlighting //
        //
        updateHighlightingState();
    }

    if (changes & DataElement::CDE_ChildSelectionChange)
    {
        updateHighlightingState();
    }
}

//################//
// SLOTS          //
//################//

void
GraphElement::hideGraphElement()
{
    if (dataElement_)
    {
        QList<DataElement *> elements;
        elements.append(dataElement_);

        HideDataElementCommand *command = new HideDataElementCommand(elements, NULL);
        getProjectGraph()->executeCommand(command);
    }
}

void
GraphElement::hideRoads()
{
    hideGraphElement();
}

//################//
// EVENTS         //
//################//

/*!
* Handles Item Changes.
*/
QVariant
GraphElement::itemChange(GraphicsItemChange change, const QVariant &value)
{
    if (change == QGraphicsItem::ItemSelectedHasChanged && dataElement_)
    {
        if (value.toBool())
        {
            // Create command only if there is a change //
            //
            if (!dataElement_->isElementSelected())
            {
                SelectDataElementCommand *command = new SelectDataElementCommand(dataElement_, NULL);
                getProjectGraph()->executeCommand(command);
                // DO NOT LET THE OBSERVER CALL itemChange WHEN YOU ARE ALREADY IN IT!
            }
        }
        else
        {
            // Create command only if there is a change //
            //
            if (dataElement_->isElementSelected())
            {
                DeselectDataElementCommand *command = new DeselectDataElementCommand(dataElement_, NULL);
                getProjectGraph()->executeCommand(command);
                // DO NOT LET THE OBSERVER CALL itemChange WHEN YOU ARE ALREADY IN IT!
            }
        }

        // Highlighting //
        //
        updateHighlightingState();
    }

    // Let the baseclass handle it //
    //
    // return QGraphicsItem::itemChange(change, value); // The default implementation does nothing, and returns value.
    return value;
}

void
GraphElement::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
    getProjectGraph()->postponeGarbageDisposal();
    contextMenu_->exec(event->screenPos());
    getProjectGraph()->finishGarbageDisposal();
}

void
GraphElement::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    // Rightclick => contextMenu, but not deselect //
    //
    if (event->button() != Qt::LeftButton)
    {
        return; // prevent deselection by doing nothing
    }

    getProjectGraph()->postponeGarbageDisposal();
    QGraphicsPathItem::mousePressEvent(event); // pass to baseclass
    getProjectGraph()->finishGarbageDisposal();
}

void
GraphElement::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    setHovered(true);

    QGraphicsPathItem::hoverEnterEvent(event); // pass to baseclass
}

void
GraphElement::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setHovered(false);

    QGraphicsPathItem::hoverLeaveEvent(event); // pass to baseclass
}
