/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <QDebug>

#include "MEIntVectorPort.h"
#include "MELineEdit.h"
#include "MEMessageHandler.h"
#include "nodes/MENode.h"

MEIntVectorPort::MEIntVectorPort(MENode *node, QGraphicsScene *scene,
                                 const QString &portname,
                                 const QString &paramtype,
                                 const QString &description)
    : MEVectorPort(node, scene, portname, paramtype, description)
{
}

MEIntVectorPort::MEIntVectorPort(MENode *node, QGraphicsScene *scene,
                                 const QString &portname,
                                 int paramtype,
                                 const QString &description,
                                 int porttype)
    : MEVectorPort(node, scene, portname, paramtype, description, porttype)
{
}

MEIntVectorPort::~MEIntVectorPort()
{
}

//------------------------------------------------------------------------
// get the float vector value
//------------------------------------------------------------------------
void MEIntVectorPort::textCB(const QString &)
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
        qCritical() << "did not find IntVector line edit";

#ifdef YAC

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;
    sb << m_nVect;

    for (int i = 0; i < m_nVect; i++)
        sb << m_vectorList[type].at(i)->text().toInt();

    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    m_vector.clear();
    for (int i = 0; i < m_nVect; i++)
    {
        QVariant v(m_vectorList[type].at(i)->text());
        m_vector.append(v);
    }
    sendParamMessage();
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

#ifdef YAC

//------------------------------------------------------------------------
void MEIntVectorPort::setValues(covise::coRecvBuffer &tb)
//------------------------------------------------------------------------
{
    tb >> m_nVect;

    int ff;
    m_vector.clear();

    for (int j = 0; j < m_nVect; j++)
    {
        tb >> ff;
        QVariant v(ff);
        m_vector.append(v);
    }

    // modify module & control line content

    if (!m_vectorList[MODULE].isEmpty())
    {
        for (int j = 0; j < m_nVect; j++)
            m_vectorList[MODULE].at(j)->setText(m_vector.at(j).toString());
    }

    if (!m_vectorList[CONTROL].isEmpty())
    {
        for (int j = 0; j < m_nVect; j++)
            m_vectorList[CONTROL].at(j)->setText(m_vector.at(j).toString());
    }
}
#endif
