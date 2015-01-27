/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QPushButton>
#include <QVBoxLayout>
#include <QDialog>
#include <QDialogButtonBox>
#include <QAction>
#include <QDebug>

#include "MEColormapChoicePort.h"
#include "MELineEdit.h"
#include "MEExtendedPart.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"
#include "color/MEColorMap.h"
#include "widgets/MEUserInterface.h"
#include "nodes/MENode.h"
#include "controlPanel/MEControlParameter.h"
#include "controlPanel/MEControlParameterLine.h"

/*!
    \class MEColormapChoicePort
    \brief handles module parameter of type colormapChoice
*/

//!-------------------------------------------------------------------------
//!
//!-------------------------------------------------------------------------
MEColormapChoicePort::MEColormapChoicePort(MENode *node, QGraphicsScene *scene,
                                           const QString &portname,
                                           const QString &paramtype,
                                           const QString &description)
    : MEParameterPort(node, scene, portname, paramtype, description)
    , m_colorMap(NULL)
{
    m_comboBox[MODULE] = m_comboBox[CONTROL] = NULL;
}

//!-------------------------------------------------------------------------
//!
//!-------------------------------------------------------------------------
MEColormapChoicePort::MEColormapChoicePort(MENode *node, QGraphicsScene *scene,
                                           const QString &portname,
                                           int paramtype,
                                           const QString &description,
                                           int porttype)
    : MEParameterPort(node, scene, portname, paramtype, description, porttype)
    , m_colorMap(NULL)
{
    m_comboBox[MODULE] = m_comboBox[CONTROL] = NULL;
}

//!-------------------------------------------------------------------------
//!
//!-------------------------------------------------------------------------
MEColormapChoicePort::~MEColormapChoicePort()
{

#ifndef YAC
    delete m_colorMap;
    m_colorMap = NULL;
#endif
}

void MEColormapChoicePort::restoreParam()
{
    m_currentChoice = m_currentChoiceold;
    m_choiceValues = m_choiceValuesold;
    sendParamMessage();
}

void MEColormapChoicePort::storeParam()
{
    m_currentChoiceold = m_currentChoice;
    m_choiceValuesold = m_choiceValues;
}

//!-------------------------------------------------------------------------
//! module has requested parameter
//!-------------------------------------------------------------------------
void MEColormapChoicePort::moduleParameterRequest()
{
    sendParamMessage();
}

//!-------------------------------------------------------------------------
//! define one parameter
//!-------------------------------------------------------------------------
void MEColormapChoicePort::defineParam(QString value, int apptype)
{

#ifdef YAC

    Q_UNUSED(value);
    Q_UNUSED(apptype);

#else

    QString values = value.trimmed();
    // create a new colormap
    m_colorMap = new MEColorMap(this, 0);
    m_colorMap->hide();

    // define  colormap & choices
    QStringList list = value.split(' ', QString::SkipEmptyParts);

    int ie = 0;
    m_currentChoice = list[ie].toInt() - 1;
    ie++;
    m_noOfChoices = list[ie].toInt();
    ie++;

    // replace '177' to allow blanks in choice label
    // store choices inside a list
    for (int j = 0; j < m_noOfChoices; j++)
    {
        QString tmp = list[ie];
        ie++;
        QString text = tmp.replace(QChar('\177'), " ");
        m_choiceValues.append(text);

        int ll = list[ie].toInt();
        ie++;
        m_mapPoints.append(ll);

        float *data = new float[ll * 5];
        for (int m = 0; m < ll * 5; m++)
        {
            data[m] = list[ie].toFloat();
            ie++;
        }
        m_values.append(data);
    }

    m_colorMap->updateColorMap(m_mapPoints[m_currentChoice], m_values[m_currentChoice]);

    MEParameterPort::defineParam(value, apptype);
#endif
}

//!-------------------------------------------------------------------------
//! modify one parameter
//!-------------------------------------------------------------------------
void MEColormapChoicePort::modifyParam(QStringList list, int noOfValues, int istart)
{
    Q_UNUSED(list);
    Q_UNUSED(noOfValues);
    Q_UNUSED(istart);
}

//!-------------------------------------------------------------------------
//! modify one parameter
//!-------------------------------------------------------------------------
void MEColormapChoicePort::modifyParameter(QString lvalue)
{
#ifdef YAC

    Q_UNUSED(lvalue);

#else
    // clear old lists
    m_choiceValues.clear();
    m_mapPoints.clear();
    m_values.clear();

    // define  new olormap & choices
    QStringList list = lvalue.split(' ', QString::SkipEmptyParts);
    int count = list.count();

    // check
    if (count <= 4)
    {
        QString text = "_________ ATTENTION: " + node->getTitle() + "::" + portname + ";  No. of values too short";
        MEUserInterface::instance()->printMessage(text);
        return;
    }

    int ie = 0;
    m_currentChoice = list[ie].toInt() - 1;
    ie++;
    m_noOfChoices = list[ie].toInt();
    ie++;

    // replace '177' to allow blanks in label
    // store choices inside a list
    for (int j = 0; j < m_noOfChoices; j++)
    {
        // name for colormap
        QString tmp = list[ie];
        ie++;
        QString text = tmp.replace(QChar('\177'), " ");

        // number of RGBAX points
        int ll = list[ie].toInt();
        ie++;
        if (ie + (ll * 5) <= count)
        {
            // points definition
            float *data = new float[ll * 5];
            for (int m = 0; m < ll * 5; m++)
            {
                data[m] = list[ie].toFloat();
                ie++;
            }

            m_mapPoints.append(ll);
            m_choiceValues.append(text);
            m_values.append(data);
        }
        else
        {
            QString text = "_________ ATTENTION: " + node->getTitle() + "::" + portname + ";  No. of values too short";
            MEUserInterface::instance()->printMessage(text);
            break;
        }
    }

    if (m_comboBox[MODULE])
    {
        m_comboBox[MODULE]->clear();
        m_comboBox[MODULE]->addItems(m_choiceValues);
        m_comboBox[MODULE]->setCurrentIndex(m_currentChoice);
    }

    if (m_comboBox[CONTROL])
    {
        m_comboBox[CONTROL]->clear();
        m_comboBox[CONTROL]->addItems(m_choiceValues);
        m_comboBox[CONTROL]->setCurrentIndex(m_currentChoice);
    }

    m_colorMap->updateColorMap(m_mapPoints[m_currentChoice], m_values[m_currentChoice]);

#endif
}

void MEColormapChoicePort::sendParamMessage()
//!-------------------------------------------------------------------------
//! send a PARAM message to controller
//!-------------------------------------------------------------------------
{
    MEParameterPort::sendParamMessage(makeColorMapValues());
}

//!-------------------------------------------------------------------------
//!
//!-------------------------------------------------------------------------
QString MEColormapChoicePort::makeColorMapValues()
{
    QStringList list;
    list << QString::number(m_currentChoice + 1);
    list << QString::number(m_noOfChoices);

    for (int k = 0; k < m_noOfChoices; k++)
    {
        list << m_choiceValues[k];
        list << QString::number(m_mapPoints[k]);

        int iend = m_mapPoints[k] * 5;
        float *data = m_values[k];
        for (int i = 0; i < iend; i++)
            list << QString::number(data[i]);
    }

    QString tmp = list.join(" ");
    return tmp;
}

//!-------------------------------------------------------------------------
//! make the COLORMAP layout
//!-------------------------------------------------------------------------
void MEColormapChoicePort::makeLayout(layoutType type, QWidget *w)
{

    QHBoxLayout *hBox = new QHBoxLayout(w);
    hBox->setMargin(2);
    hBox->setSpacing(2);

    m_comboBox[type] = new MEComboBox();
    m_comboBox[type]->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
    m_comboBox[type]->addItems(m_choiceValues);
    m_comboBox[type]->setCurrentIndex(m_currentChoice);
    connect(m_comboBox[type], SIGNAL(activated(int)), this, SLOT(choiceCB(int)));
    connect(m_comboBox[type], SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));

    hBox->addWidget(m_comboBox[type]);
    hBox->addStretch(0);

    // add combobox with predefined colormaps & m_preview
    if (type == MODULE)
        m_preview[type] = m_colorMap->getModulePreview();
    else
        m_preview[type] = m_colorMap->getControlPreview();

    m_preview[type]->show();
    hBox->addWidget(m_preview[type], 1);
}

//!
//! Get the new choice value
//!
void MEColormapChoicePort::choiceCB(int state)
{

#ifdef YAC

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;
    sb << 0 << state;
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    m_currentChoice = state;
    sendParamMessage();
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

#ifdef YAC
void MEColormapChoicePort::setValues(covise::coRecvBuffer &)
{
}
#endif
