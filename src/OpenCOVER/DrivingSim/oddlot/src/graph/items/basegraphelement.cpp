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

#include "basegraphelement.hpp"

// Data //
//
#include "src/data/dataelement.hpp"
#include "src/data/commands/dataelementcommands.hpp"

// Graph //
//
#include "src/graph/profilegraph.hpp"
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

#include <QtSvg/QGraphicsSvgItem>

//################//
// CONSTRUCTOR    //
//################//


template<class T>
BaseGraphElement<T>::BaseGraphElement(BaseGraphElement<T> *parentBaseGraphElement, DataElement *dataElement)
    : T(parentBaseGraphElement)
    , Observer()
    , parentBaseGraphElement_(parentBaseGraphElement)
    , dataElement_(dataElement)
    , hovered_(false)
    , useHighlighting_(true)
    , highlightOpacity_(1.0)
    , normalOpacity_(0.5)
    , isInGarbage_(false)
{
    init();
}


template<class T>
BaseGraphElement<T>::~BaseGraphElement()
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

template<class T>
QGraphicsItem *
BaseGraphElement<T>::This()
{
    return static_cast<QGraphicsItem *>(this);
}

template<class T>
const QGraphicsItem *
BaseGraphElement<T>::This() const
{
    return static_cast<const QGraphicsItem *>(this);
}

template<class T>
void
BaseGraphElement<T>::init()
{
    updateHighlightingState();

    // ContextMenu //
    //
    contextMenu_ = new QMenu();

    hideMenu_ = new QMenu(QObject::tr("Hide"));
    contextMenu_->addMenu(hideMenu_);

    removeMenu_ = new QMenu(QObject::tr("Delete"));
    contextMenu_->addMenu(removeMenu_);

    getContextMenu()->addSeparator();

    // Selection //
    //
    This()->setFlag(QGraphicsItem::ItemIsSelectable, false); // not selectable by default

    if (dataElement_)
    {
        // Selection/Hiding //
        //
        This()->setVisible(!dataElement_->isElementHidden());
        This()->setCacheMode(QGraphicsItem::DeviceCoordinateCache);
        //setSelected(dataElement_->isElementSelected()); // will be ignored, if not selectable

        // Observer //
        //
        dataElement_->attachObserver(this);
    }
}

template<class T>
void
BaseGraphElement<T>::setSelectable()
{
    if (dataElement_)
    {
        This()->setFlag(QGraphicsItem::ItemIsSelectable, true);
        This()->setSelected(dataElement_->isElementSelected());
    }
}

template<class T>
void
BaseGraphElement<T>::registerForDeletion()
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

template<class T>
void
BaseGraphElement<T>::notifyDeletion()
{
    isInGarbage_ = true;
}

template<class T>
ProjectData *
BaseGraphElement<T>::getProjectData() const
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

template<class T>
ProjectGraph *
BaseGraphElement<T>::getProjectGraph() const
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
template<class T>
TopviewGraph *
BaseGraphElement<T>::getTopviewGraph() const
{
    if (parentBaseGraphElement_)
    {
        return parentBaseGraphElement_->getTopviewGraph();
    }
    else
    {
        return NULL;
    }
}

template<class T>
ProfileGraph *
BaseGraphElement<T>::getProfileGraph() const
{
    if (parentBaseGraphElement_)
    {
        return parentBaseGraphElement_->getProfileGraph();
    }
    else
    {
        return NULL;
    }
}

//################//
// BaseGraphElement   //
//################//

template<class T>
void
BaseGraphElement<T>::setHovered(bool hovered)
{
    if (hovered_ != hovered)
    {
        hovered_ = hovered;
        updateHighlightingState();
    }
    if (parentBaseGraphElement_)
    {
        parentBaseGraphElement_->setHovered(hovered);
    }
}

template<class T>
void
BaseGraphElement<T>::enableHighlighting(bool enable)
{
    useHighlighting_ = enable;
}

template<class T>
void
BaseGraphElement<T>::setHighlighting(bool highlight)
{
    if (!useHighlighting_)
    {
        return;
    }

    if (highlight)
    {
        This()->setOpacity(highlightOpacity_);
    }
    else
    {
        This()->setOpacity(normalOpacity_);
    }
}

template<class T>
void
BaseGraphElement<T>::setOpacitySettings(double highlightOpacity, double normalOpacity)
{
    highlightOpacity_ = highlightOpacity;
    normalOpacity_ = normalOpacity;
    updateHighlightingState();
}

template<class T>
void
BaseGraphElement<T>::updateHighlightingState()
{
    setHighlighting(hovered_ || (dataElement_ && (dataElement_->isElementSelected() || dataElement_->isChildElementSelected())));
}

//################//
// OBSERVER       //
//################//

template<class T>
void
BaseGraphElement<T>::updateObserver()
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
        This()->setVisible(!dataElement_->isElementHidden()); // Do this before handling the selection!
    }

    // Selection //
    //
    if (changes & DataElement::CDE_SelectionChange)
    {
        // Selection //
        //
        if (This()->isSelected() != dataElement_->isElementSelected())
        {
            // DO NOT LET THE OBSERVER CALL itemChange() WHEN YOU ARE ALREADY IN IT!
            This()->setSelected(dataElement_->isElementSelected());
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
// EVENTS         //
//################//

/*!
* Handles Item Changes.
*/
template<class T>
QVariant
BaseGraphElement<T>::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
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

template<class T>
void
BaseGraphElement<T>::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
    getProjectGraph()->postponeGarbageDisposal();
    contextMenu_->exec(event->screenPos());
    getProjectGraph()->finishGarbageDisposal();
}

template<class T>
void
BaseGraphElement<T>::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    // Rightclick => contextMenu, but not deselect //
    //
    if (event->button() != Qt::LeftButton)
    {
        return; // prevent deselection by doing nothing
    }

    getProjectGraph()->postponeGarbageDisposal();
    T::mousePressEvent(event); // pass to baseclass
    getProjectGraph()->finishGarbageDisposal();
}

template<class T>
void
BaseGraphElement<T>::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    setHovered(true);

    T::hoverEnterEvent(event); // pass to baseclass
}

template<class T>
void
BaseGraphElement<T>::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setHovered(false);

    T::hoverLeaveEvent(event); // pass to baseclass
}

template class BaseGraphElement<QGraphicsPathItem>;  // Explicit instantiation
template class BaseGraphElement<QGraphicsSvgItem>;
