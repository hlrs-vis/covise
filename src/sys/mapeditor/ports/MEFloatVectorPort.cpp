/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QDebug>

#include "MEFloatVectorPort.h"
#include "MELineEdit.h"
#include "MEMessageHandler.h"
#include "nodes/MENode.h"

;

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
MEFloatVectorPort::MEFloatVectorPort(MENode *node, QGraphicsScene *scene,
                                     const QString &portname,
                                     const QString &paramtype,
                                     const QString &description)
    : MEVectorPort(node, scene, portname, paramtype, description)
{
}

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
MEFloatVectorPort::MEFloatVectorPort(MENode *node, QGraphicsScene *scene,
                                     const QString &portname,
                                     int paramtype,
                                     const QString &description,
                                     int porttype)
    : MEVectorPort(node, scene, portname, paramtype, description, porttype)
{
}

//------------------------------------------------------------------------
MEFloatVectorPort::~MEFloatVectorPort()
//------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------
// get the float vector value
//------------------------------------------------------------------------
void MEFloatVectorPort::textCB(const QString &)
{

    // object that sent the signal

    const QObject *obj = sender();
    MELineEdit *le = (MELineEdit *)obj;

    // find widget that send the  in list

    layoutType type = MODULE;
    if (m_vectorList[MODULE].contains(le))
        type = MODULE;

    else if (m_vectorList[CONTROL].contains(le))
        type = CONTROL;

    else
        qCritical() << "did not find FloatVector line edit" << endl;


    m_vector.clear();
    for (int i = 0; i < m_nVect; i++)
    {
        QVariant v(m_vectorList[type].at(i)->text());
        m_vector.append(v);
    }
    sendParamMessage();

    // inform parent widget that value has been changed
    node->setModified(true);
}

