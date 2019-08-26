/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QGraphicsScene>
#include <QDebug>
#include <QStatusBar>

#include "MEDataPort.h"
#include "MEMessageHandler.h"
#include "widgets/MEUserInterface.h"
#include "widgets/MEGraphicsView.h"
#include "handler/MEMainHandler.h"
#include "handler/MEPortSelectionHandler.h"
#include "dataObjects/MEDataTree.h"
#include "nodes/MENode.h"

#include <do/coDoSet.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPoints.h>
#include <do/coDoSpheres.h>
#include <do/coDoLines.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoTexture.h>
#include <do/coDoDoubleArr.h>

#include <covise/covise_msg.h>

#include <do/coDoColormap.h>
#include <do/coDoGeometry.h>

;

static bool isRegularType(const QString &t);
static bool isKnownType(const QString &t);

/*!
   \class MEDataPort
   \brief Class handles data ports

   Subclassed from MEPort
*/

MEDataPort::MEDataPort(MENode *node, QGraphicsScene *scene,
                       const QString &pname,
                       const QString &description,
                       int ptype)
    : MEPort(node, scene, pname, description, ptype)
    , synced(false)
    , treeRoot(NULL)
    , m_dataObject(NULL)
{
}

MEDataPort::MEDataPort(MENode *node, QGraphicsScene *scene,
                       const QString &pname,
                       const QString &dtypes,
                       const QString &description,
                       const QString &demand,
                       int ptype,
                       bool sync)
    : MEPort(node, scene, pname, description, ptype)
    , synced(sync)
    , treeRoot(NULL)
    , m_dataObject(NULL)
{
    setDemandType(demand);
    setDataTypes(dtypes);
}

MEDataPort::~MEDataPort()
{
}

//!
//! context menu requested
//!
void MEDataPort::contextMenuEvent(QGraphicsSceneContextMenuEvent *e)
{
    if (!synced && isConnectable())
        MEGraphicsView::instance()->showPossiblePorts(this, e);
}

//!
//!  highlight connection lines & ports
//!
void MEDataPort::hoverEnterEvent(QGraphicsSceneHoverEvent *e)
{
    MEUserInterface::instance()->statusBar()->showMessage(m_helpText);
    MEPort::hoverEnterEvent(e);
}

//!
//! reset connection lines & ports
//!
void MEDataPort::hoverLeaveEvent(QGraphicsSceneHoverEvent *e)
{
    MEUserInterface::instance()->statusBar()->clearMessage();
    MEPort::hoverLeaveEvent(e);
}

//!
//! add a link
//!
void MEDataPort::addLink(MEPort *connectedPort)
{
    // add link number

    links++;

    // store connected port & set the corresponding port object name

    connectedPorts.append(static_cast<MEDataPort *>(connectedPort));
    if (portType == DIN)
    {
        m_objectPortName = static_cast<MEDataPort *>(connectedPort)->getPortObjectName();
        setHelpText();
    }
}

//!
//! remove a link
//!
void MEDataPort::delLink(MEPort *connectedPort)
{
    // reduce link number

    links--;

    // remove connected port & set the corresponding port to NONE

    if (portType == DIN && links == 0)
    {
        m_objectPortName = "NONE";
        setHelpText();
    }
    connectedPorts.remove(connectedPorts.indexOf(static_cast<MEDataPort *>(connectedPort)));
}

//!
//! create a tree root for this port inside the data tree
//!
MEDataTreeItem *MEDataPort::createTreeRoot()
{

    QString tmp = portname + ":: " + m_objectPortName;
    if (treeRoot == NULL && node->getTreeItem() != NULL)
    {
        treeRoot = new MEDataTreeItem(node->getTreeItem(), tmp, this);

        // create a dummy object so that user can see a normal tree item to open
        new MEDataTreeItem(treeRoot, "dummy");
    }

    treeRoot->setText(0, tmp);
    treeRoot->updateItem();
    return treeRoot;
}

//!
//! retrieve the current data object for an output port
//!
void MEDataPort::updateDataObjectNames(const QString &name)
{

    m_objectPortName = name;

    if (node->isCloned())
        return;

    // get data object for local output ports

    getDataObjectInfo();

    // create/update data tree root item

    createTreeRoot();
}

//!
//! get information about the type of a distributes data object
//! show data output port with another rectangle color
//!
void MEDataPort::getDataObjectInfo()
{

    if (m_dataObject)
    {
        delete m_dataObject;
        m_dataObject = NULL;
    }

    // show only data objects on local host & for output ports that are connected to a local input port

    m_dataObjectName = "n.a.";
    m_simpleDataTypeName = "n.a.";

    bool lookup = false;
    if (node->isLocalNode())
        lookup = true;

    else
    {
        foreach (MEDataPort *dp, connectedPorts)
        {
            if (dp->getNode()->isLocalNode())
                lookup = true;
        }
    }

    if (lookup)
    {
        m_dataObject = covise::coDistributedObject::createFromShm(covise::coObjInfo(m_objectPortName.toLatin1().data()));
        m_dataObjectName = getDataObjectString(m_dataObject);

        // check if actual data object type is a member of the allowed data type list
        m_simpleDataTypeName = getDataObjectString(m_dataObject,
                                                   true /*recurse*/, true /*only leaves*/);

        hasObjData = false;
        if (m_dataObject && m_simpleDataTypeName != "Empty Set")
        {
            if (!m_datatypes.contains(m_simpleDataTypeName)
                && !m_datatypes.contains("coDistributedObject")
                && (m_simpleDataTypeName == "Float" && !m_datatypes.contains("MinMax_Data")))
            {
                qCritical() << getNode()->getName() << portname << ":: Created dataobject " << m_simpleDataTypeName << " is not in allowed datatype list " << m_datatypes;
            }
            else if (portType == DOUT)
            {
                hasObjData = true;
            }
        }
        update();
    }
}

//!
//! get the name of the distributed data object
//!
QString MEDataPort::getDataNameList()
{
    if (portType == MEPort::DIN && !connectedPorts.isEmpty())
    {
        QStringList objs;
        foreach (MEDataPort *pp, connectedPorts)
        {
            objs << pp->getDataObjectName();
        }
        return objs.join(" ");
    }

    else
        return m_dataObjectName;
}

//!
//! generate a tooltip for the data object port
//!
void MEDataPort::setHelpText()
{
    if (!m_dataObject)
        getDataObjectInfo();

    m_helpText = m_datatypes.join(", ");
    QString tip = description + " <i>(" + m_dataObjectName + ")</i>";
    setToolTip(tip);
}

//!
//! update  a tooltip for the data object port
//!
void MEDataPort::updateHelpText(const QString &text)
{
    m_helpText = text;
}

//!
//! show content of data port in the data viewer
//!
void MEDataPort::showDataContent()
{
    MEUserInterface::instance()->showDataViewer(true);
    MEDataTree::instance()->clearSelection();
    treeRoot->parent()->setExpanded(true);
    treeRoot->setSelected(true);
}

//!
//! copy a data object
//!
void MEDataPort::copyDataObject()
{
    foreach (MEDataTreeItem *nptr, treeList)
    {
        if (MEDataTree::instance() && MEDataTree::instance()->lostfound)
        {
            MEDataTreeItem *it = new MEDataTreeItem(MEDataTree::instance()->lostfound, nptr->text(0));
            for (int i = 1; i < 6; i++)
                it->setText(i, nptr->text(i));
        }
    }
}

//!
//! remove a data object
//!
void MEDataPort::removeDataObject(int id1, int id2, int id3,
                                  int block, int timestep)
{
    QString d1, d2, d3, d4, d5;

    d1.setNum(id1);
    d2.setNum(id2);
    d3.setNum(id3);
    d4.setNum(block);
    d5.setNum(timestep);

    foreach (MEDataTreeItem *nptr, treeList)
    {
        // I had problems compiling that impression with
        // llvm. clang complains, that using the ,(comma)
        // operator, the return value of the first operator
        // is simply ignored
        if (/*!nptr->text(1).isEmpty(), */ !d1.isEmpty() &&
            /*!nptr->text(2).isEmpty(), */ !d3.isEmpty() && // btw, was the intention to query d2 rather than d3?
            /*!nptr->text(3).isEmpty(), */ !d3.isEmpty() &&
            /*!nptr->text(4).isEmpty(), */ !d4.isEmpty() &&
            /*!nptr->text(5).isEmpty(), */ !d5.isEmpty())
        {
            treeList.remove(treeList.indexOf(nptr));
            delete nptr;
            break;
        }
    }
}

//!
//! set the identifier for a data object (timestep, blocknumer..)
//!
void MEDataPort::setDataObject(int id1, int id2, int id3, int block, int timestep, QString name)
{
    m_objectPortName = name;
    MEDataTreeItem *tree = new MEDataTreeItem(node->getTreeItem(), m_objectPortName);

    tree->setText(1, QString::number(id1));
    tree->setText(2, QString::number(id2));
    tree->setText(3, QString::number(id3));
    tree->setText(4, QString::number(block));
    tree->setText(5, QString::number(timestep));

    treeList << tree;
}

//!
//! possible data types for this data port
//!
void MEDataPort::setDataTypes(const QString &connections)
{
    m_datatypes = connections.split("|", QString::SkipEmptyParts);
    m_datatypes.sort();

    for (int i = 0; i < m_datatypes.size() - 1; ++i)
    {
        if (m_datatypes[i] == m_datatypes[i + 1])
            qDebug() << "module " << node->getName() << " contains duplicate " << m_datatypes[i] << " at port " << getName();
    }

    m_specialdatatypes.clear();
    foreach (QString dt, m_datatypes)
    {
        if (!isKnownType(dt))
            qDebug() << "module " << node->getName() << " handles unknown type " << dt << " at port " << getName();

        if (!isRegularType(dt))
            m_specialdatatypes << dt;
    }

    // create a tooltip for this port
    setHelpText();
}

//!
//! define the demand type of a data port
//!
void MEDataPort::setDemandType(const QString &dtype)
{
    // set the right port color
    if (dtype == "req")
        demand = REQ;

    else if (dtype == "default")
        demand = DEF;

    else if (dtype == "opt")
        demand = OPT;

    else if (dtype.contains("dep") != 0)
    {
        QStringList list = dtype.split(" ", QString::SkipEmptyParts);
        demand = DEP;
        dependency = list[1];
    }

    // set port color
    portcolor = definePortColor();
    int h, s, v;
    portcolor.getHsv(&h, &s, &v);
    portcolor_dark.setHsv(h, 255, v);
    setBrush(portcolor);
    setPen(portcolor_dark);
}

//!
//! get information about the type of a data object
//!

QString MEDataPort::getDataObjectString(const covise::coDistributedObject *obj, bool recurse, bool onlyLeaf)
{
    if (!obj)
    {
        return QString("No data");
    }

    if (const covise::coDoSet *set = dynamic_cast<const covise::coDoSet *>(obj))
    {
        if (recurse)
        {
            int no_elem;
            const covise::coDistributedObject *const *elem = set->getAllElements(&no_elem);
            if (no_elem == 0)
                return QString("Empty Set");

            QStringList elemTypes;
            for (int i = 0; i < no_elem; i++)
            {
                QString type = getDataObjectString(elem[i], true, onlyLeaf);
                if (!elemTypes.contains(type))
                    elemTypes << type;
            }

            if (onlyLeaf)
            {
                return elemTypes[0];
            }
            else
            {
                bool braces = (elemTypes.size() > 1);
                QString res = elemTypes.join(", ");
                if (braces)
                    return "Set of (" + res + ")";
                else
                    return "Set of " + res;
            }
        }
        else
            return QString("Set");
    }

    else if (dynamic_cast<const covise::coDoUniformGrid *>(obj))
        return QString("UniformGrid");

    else if (dynamic_cast<const covise::coDoRectilinearGrid *>(obj))
        return QString("RectilinearGrid");

    else if (dynamic_cast<const covise::coDoStructuredGrid *>(obj))
        return QString("StructuredGrid");

    else if (dynamic_cast<const covise::coDoUnstructuredGrid *>(obj))
        return QString("UnstructuredGrid");

    else if (dynamic_cast<const covise::coDoPoints *>(obj))
        return QString("Points");

    else if (dynamic_cast<const covise::coDoLines *>(obj))
        return QString("Lines");

    else if (dynamic_cast<const covise::coDoPolygons *>(obj))
        return QString("Polygons");

    else if (dynamic_cast<const covise::coDoTriangleStrips *>(obj))
        return QString("TriangleStrips");

    else if (dynamic_cast<const covise::coDoTriangles *>(obj))
        return QString("Triangles");

    else if (dynamic_cast<const covise::coDoQuads *>(obj))
        return QString("Quads");

    else if (dynamic_cast<const covise::coDoRGBA *>(obj))
        return QString("RGBA");

    else if (dynamic_cast<const covise::coDoTexture *>(obj))
        return QString("Texture");

    else if (dynamic_cast<const covise::coDoFloat *>(obj))
        return QString("Float");

    else if (dynamic_cast<const covise::coDoVec2 *>(obj))
        return QString("Vec2");

    else if (dynamic_cast<const covise::coDoVec3 *>(obj))
        return QString("Vec3");

    else if (dynamic_cast<const covise::coDoSpheres *>(obj))
        return QString("Spheres");

    else if (dynamic_cast<const covise::coDoMat3 *>(obj))
        return QString("Mat3");

    else if (dynamic_cast<const covise::coDoTensor *>(obj))
        return QString("Tensor");

    else if (dynamic_cast<const covise::coDoByte *>(obj))
        return QString("Byte");

    else if (dynamic_cast<const covise::coDoInt *>(obj))
        return QString("Int");

    else if (dynamic_cast<const covise::coDoColormap *>(obj))
        return QString("ColorMap");

    else if (dynamic_cast<const covise::coDoGeometry *>(obj))
        return QString("Geometry");

    else if (dynamic_cast<const covise::coDoIntArr *>(obj))
        return QString("IntArr");
    else if (dynamic_cast<const covise::coDoDoubleArr *>(obj))
        return QString("DoubleArr");

    return QString("(unknown type)");
}

static bool isRegularType(const QString &t)
{
    static QStringList regularTypes;
    if (regularTypes.isEmpty())
    {
        regularTypes
            << "UniformGrid"
            << "RectilinearGrid"
            << "StructuredGrid"
            << "UnstructuredGrid"
            << "Points"
            << "Lines"
            << "Polygons"
            << "Quads"
            << "Triangles"
            << "TriangleStrips"
            << "RGBA"
            << "PixelImage"
            << "Texture"
            << "Float"
            << "Byte"
            << "Int"
            << "Vec2"
            << "Vec3"
            << "Mat3"
            << "Tensor"
            << "Spheres"
            << "ColorMap"
            << "Geometry"
            << "IntArr"
            << "Text"
            << "OctTree"
            << "OctTreeP"
            << "DoubleArr";
    }

    return regularTypes.contains(t);
}

static bool isKnownType(const QString &t)
{
    static QStringList knownTypes;
    if (knownTypes.isEmpty())
    {
        knownTypes
            << "coDistributedObject"
            << "MinMax_Data"
            << "USR_DistFenflossBoco"
            << "USR_FenflossBoco"
            << "USR_FoamBoco"
            << "USR_FoamMesh"
            << "Text_Iv"
            << "USR_AnGeo"
            << "USR_ScaFun"
            << "USR_ExtInfo";
    }
    return (isRegularType(t) || knownTypes.contains(t));
}

//!
//! set an index for a data object name
//!
void MEDataPort::setReadIndex(int p, const QString &name)
{
    index = p;
    m_objectPortName = name;
}

//!
//! send an object message message to controller
//!
void MEDataPort::sendObjectMessage(const QString &key)
{

    QStringList buffer;
    if (MEMainHandler::instance()->isMaster())
    {
        buffer << key << node->getName() << node->getNumber() << node->getHostname();
        buffer << portname << m_objectPortName;
        QString data = buffer.join("\n");

        MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
        buffer.clear();
    }
}

//!
//! get the color depending on the port type
//!
QColor MEDataPort::definePortColor()
{
    QColor col;
    col = Qt::black;

    switch (demand)
    {
    case REQ:
        col = MEMainHandler::s_requestedColor;
        break;

    case DEF:
        col = MEMainHandler::s_defColor;
        break;

    case OPT:
        col = MEMainHandler::s_optionalColor;
        break;

    case DEP:
        col = MEMainHandler::s_dependentColor;
        break;
    }

    return (col);
}

bool MEDataPort::arePortsCompatible(MEDataPort *pout, MEDataPort *pin, bool ignorePossibleTypes)
{
    if (pout->getPortType() != MEPort::DOUT)
        std::swap(pin, pout);

    if (pout->getPortType() != MEPort::DOUT)
        return false;

    if (pin->getPortType() != MEPort::DIN && pin->getPortType() != MEPort::MULTY_IN)
        return false;

    QStringList tin = pin->getDataTypes();
    if (tin.contains("coDistributedObject"))
        return true;

    QStringList tout;
    QString dobjn = pout->getSimpleDataTypeName();
    if (dobjn == "No data" || dobjn == "Empty Set" || dobjn == "n.a.")
        tout << pout->getDataTypes();
    else if (!ignorePossibleTypes)
    {
        tout << pout->getDataTypes();
        if (!tout.contains(dobjn))
        {
            if ((!(dobjn == "Float" || dobjn == "IntArr")) || !tout.contains("MinMax_Data"))
            {
                qDebug() << "module " << pout->node->getName() << " produces unannounced type " << dobjn << " at port " << pout->getName();
                tout << dobjn;
            }
        }
    }
    else
    {
        tout << dobjn;
        tout << pout->getSpecialDataTypes();
    }

    if (tout.contains("coDistributedObject"))
        return true;

    foreach (QString out, tout)
    {
        if (tin.contains(out))
            return true;
    }

    return false;
}
