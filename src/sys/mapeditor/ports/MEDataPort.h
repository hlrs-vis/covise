/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_DATAPORT_H
#define ME_DATAPORT_H

#include <QVector>

#include "ports/MEPort.h"

namespace covise
{
class coDistributedObject;
}
class MENode;
class MEDataTreeItem;
class MEDataPort;

//================================================
class MEDataPort : public MEPort
//================================================
{

    Q_OBJECT

public:
    MEDataPort(MENode *node, QGraphicsScene *scene, const QString &pname, const QString &description, int type = UNKNOWN);
    MEDataPort(MENode *node, QGraphicsScene *scene, const QString &pname, const QString &dtypes, const QString &description, const QString &demand,
               int type = UNKNOWN, bool sync = false);

    ~MEDataPort();

    enum demandtypes
    {
        REQ = 1,
        OPT,
        DEF,
        DEP
    };

    bool isSynced()
    {
        return synced;
    };
    int getPortIndex()
    {
        return index;
    }
    int getDemand()
    {
        return demand;
    }

    void setHelpText();
    void updateHelpText(const QString &);
    void sendObjectMessage(const QString &);
    void copyDataObject();
    void updateDataObjectNames(const QString &);
    void setDataTypes(const QString &);
    void setPortObjectName(const QString &name)
    {
        m_objectPortName = name;
    };
    void setDemandType(const QString &);
    void setDataObject(int, int, int, int, int, QString);
    void removeDataObject(int, int, int, int, int);
    void setReadIndex(int, const QString &);
    void getDataObjectInfo();

    QString getDependency()
    {
        return dependency;
    }
    QString getPortObjectName()
    {
        return m_objectPortName;
    }
    QString getDataObjectName()
    {
        return m_dataObjectName;
    }
    QString getSimpleDataTypeName()
    {
        return m_simpleDataTypeName;
    }
    QString getDataNameList();
    const QStringList &getDataTypes()
    {
        return m_datatypes;
    }
    const QStringList &getSpecialDataTypes()
    {
        return m_specialdatatypes;
    }

    const covise::coDistributedObject *getDataObject()
    {
        return m_dataObject;
    }

    MEDataTreeItem *getTreeRoot()
    {
        return treeRoot;
    }
    MEDataTreeItem *createTreeRoot();

    QVector<MEDataPort *> connectedPorts;

#ifdef YAC
    void addPortItems(covise::coRecvBuffer &);
#endif
    static QString getDataObjectString(const covise::coDistributedObject *obj, bool recurse = true, bool onlyLeaf = false);
    static bool arePortsCompatible(MEDataPort *pout, MEDataPort *pin, bool ignorePossibleTypes = true);

public slots:

    void showDataContent();

private:
    bool required, synced;
    int demand, index;

    MEDataTreeItem *treeRoot;

    const covise::coDistributedObject *m_dataObject;

    QString m_objectPortName, dependency, m_dataObjectName, m_simpleDataTypeName, m_helpText;
    QStringList m_datatypes, m_specialdatatypes;
    QVector<MEDataTreeItem *> treeList;
    QColor definePortColor();

    void addLink(MEPort *port);
    void delLink(MEPort *port);

protected:
    void contextMenuEvent(QGraphicsSceneContextMenuEvent *e);
    void hoverEnterEvent(QGraphicsSceneHoverEvent *e);
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *e);
};
#endif
