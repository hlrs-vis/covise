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

#include "MEColorMapPort.h"
#include "MELineEdit.h"
#include "MEExtendedPart.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"
#include "color/MEColorMap.h"
#include "widgets/MEUserInterface.h"
#include "nodes/MENode.h"
#include "controlPanel/MEControlParameter.h"
#include "controlPanel/MEControlParameterLine.h"

;

/*!
    \class MEColorMapPort
    \brief handles module parameter of type colormap
*/

//!-------------------------------------------------------------------------
//!
//!-------------------------------------------------------------------------
MEColorMapPort::MEColorMapPort(MENode *node, QGraphicsScene *scene,
                               const QString &portname,
                               const QString &paramtype,
                               const QString &description)
    : MEParameterPort(node, scene, portname, paramtype, description)
    , cmapSteps(0)
    , cmapRGBAX(NULL)
    , values(NULL)
    , m_fileOpen(false)
    , m_colorMap(NULL)
{
    m_extendedPart[0] = m_extendedPart[1] = NULL;
    m_dialog = NULL;
    folderAction[0] = folderAction[1] = NULL;
}

//!-------------------------------------------------------------------------
//!
//!-------------------------------------------------------------------------
MEColorMapPort::MEColorMapPort(MENode *node, QGraphicsScene *scene,
                               const QString &portname,
                               int paramtype,
                               const QString &description,
                               int porttype)
    : MEParameterPort(node, scene, portname, paramtype, description, porttype)
    , cmapSteps(0)
    , cmapRGBAX(NULL)
    , m_fileOpen(false)
    , m_colorMap(NULL)
{
    m_extendedPart[0] = m_extendedPart[1] = NULL;
    m_dialog = NULL;
    folderAction[0] = folderAction[1] = NULL;
}

//!-------------------------------------------------------------------------
//!
//!-------------------------------------------------------------------------
MEColorMapPort::~MEColorMapPort()
{

#ifndef YAC
    if (m_dialog)
    {
        delete m_dialog;
        m_dialog = NULL;
    }

    if (cmapRGBAX)
    {
        delete[] cmapRGBAX;
        cmapRGBAX = NULL;
    }
#endif
}

void MEColorMapPort::restoreParam() {}
void MEColorMapPort::storeParam() {}

//!-------------------------------------------------------------------------
//! module has requested parameter
//!-------------------------------------------------------------------------
void MEColorMapPort::moduleParameterRequest()
{
    sendParamMessage();
}

//!-------------------------------------------------------------------------
//! define one parameter
//!-------------------------------------------------------------------------
void MEColorMapPort::defineParam(QString value, int apptype)
{

#ifdef YAC

    Q_UNUSED(value);
    Q_UNUSED(apptype);

#else

    // create a new colormap
    m_colorMap = new MEColorMap(this, 0);

    // fill colormap
    QStringList list = value.split(' ', QString::SkipEmptyParts);
    if (list[2] == "RGBAX")
    {
        int numSteps = list[3].toInt();
        if (numSteps != cmapSteps)
        {
            delete[] cmapRGBAX;
            cmapRGBAX = NULL;
        }
        cmapSteps = numSteps;

        if (cmapRGBAX == NULL)
        {
            cmapRGBAX = new float[cmapSteps * 5];
        }

        for (int i = 0; i < cmapSteps * 5; i++)
        {
            cmapRGBAX[i] = list[4 + i].toFloat();
        }

        m_colorMap->updateColorMap(cmapSteps, cmapRGBAX);
    }
    m_fileOpen = false;

    MEParameterPort::defineParam(value, apptype);
#endif
}

//!-------------------------------------------------------------------------
//! modify one parameter
//!-------------------------------------------------------------------------
void MEColorMapPort::modifyParam(QStringList list, int noOfValues, int istart)
{
#ifdef YAC

    Q_UNUSED(list);
    Q_UNUSED(noOfValues);
    Q_UNUSED(istart);

#else

    // COLORMAP (0 = min, 1 = max, 2 = type, 3 = numSteps,
    //     4 = R G B A X  R G B A X  ...)

    int count = list.count();
    if (count > istart + 1)
    {
        if (count == istart + 2)
        {
            // module has set new minimun and maximun
            if (noOfValues == 2)
            {
                m_colorMap->updateMin(list[istart].toFloat());
                m_colorMap->updateMax(list[istart + 1].toFloat());
            }
            else
                qWarning() << "Invalid ColorMap description " << list[istart + 2] << endl;
        }

        else if (list[istart + 2] == "RGBAX")
        {
            int numSteps = 0;
            if (count > istart + 3)
                numSteps = list[istart + 3].toInt();

            if (numSteps != cmapSteps)
            {
                delete[] cmapRGBAX;
                cmapRGBAX = NULL;
            }

            cmapSteps = numSteps;
            if (cmapRGBAX == NULL)
                cmapRGBAX = new float[5 * cmapSteps];

            if (count > istart + cmapSteps)
            {
                for (int i = 0; i < cmapSteps; i++)
                {
                    QStringList item = list[istart + 4 + i].split(QRegExp("\\s+"), QString::SkipEmptyParts);
                    for (int j = 0; j < 5; j++)
                    {
                        if (j < item.count())
                        {
                            cmapRGBAX[i * 5 + j] = item[j].toFloat();
                        }
                    }
                }
            }

            if (MEMainHandler::instance()->isInMapLoading() || !MEMainHandler::instance()->isMaster())
                m_colorMap->updateColorMap(cmapSteps, cmapRGBAX);
        }
    }

    else
    {
        QString msg = "MEParameterPort::modifyParam: " + node->getNodeTitle() + ": Parameter type " + parameterType + " has wrong number of values";
        MEUserInterface::instance()->printMessage(msg);
    }
#endif
}

//!-------------------------------------------------------------------------
//! modify one parameter
//!-------------------------------------------------------------------------
void MEColorMapPort::modifyParameter(QString lvalue)
{
#ifdef YAC

    Q_UNUSED(lvalue);

#else

    // COLORMAP (0 = min, 1 = max, 2 = type, 3 = numSteps,
    //     4 = R G B A X  R G B A X  ....
    // HISTO  np values

    QStringList list = QString(lvalue).split(" ");
    int count = list.count();
    int ncval = list[3].toInt();

    if (count >= 2)
    {
        m_colorMap->updateMin(list[0].toFloat());
        m_colorMap->updateMax(list[1].toFloat());

        if (count > 3)
        {
            // color values
            if (list[2] == "RGBAX")
            {
                int numSteps = 0;
                if (count > 3)
                    numSteps = list[3].toInt();

                if (numSteps != cmapSteps)
                {
                    delete[] cmapRGBAX;
                    cmapRGBAX = NULL;
                }

                cmapSteps = numSteps;
                if (cmapRGBAX == NULL)
                    cmapRGBAX = new float[5 * cmapSteps];

                if (count > cmapSteps + 4)
                {
                    for (int i = 0; i < cmapSteps * 5; i++)
                        cmapRGBAX[i] = list[4 + i].toFloat();

                    if (MEMainHandler::instance()->isInMapLoading() || !MEMainHandler::instance()->isMaster())
                        m_colorMap->updateColorMap(cmapSteps, cmapRGBAX);
                }
            }

            // histogram data
            int ip = ncval * 5 + 4;
            if (count > ncval * 5 + 4 && list[ip] == "HISTO")
            {
                ip++;
                int ndval = list[ip].toInt();
                ip++;
                float xmin = list[ip].toFloat();
                ip++;
                float xmax = list[ip].toFloat();
                ip++;
                if (values)
                    delete[] values;
                values = new int[ndval];
                for (int i = 0; i < ndval - 2; i++)
                    values[i] = list[ip + i].toInt();
                m_colorMap->updateHistogram(ndval - 2, xmin, xmax, values);
            }
        }
    }

    else
    {
        QString msg = "MEParameterPort::modifyParam: " + node->getNodeTitle() + ": Parameter type " + parameterType + " has wrong number of values";
        MEUserInterface::instance()->printMessage(msg);
    }
#endif
}

void MEColorMapPort::sendParamMessage()
//!-------------------------------------------------------------------------
//! send a PARAM message to controller
//!-------------------------------------------------------------------------
{
    MEParameterPort::sendParamMessage(makeColorMapValues());
}

//!-------------------------------------------------------------------------
//!
//!-------------------------------------------------------------------------
QString MEColorMapPort::makeColorMapValues()
{
    QStringList list;

    list << QString::number(m_colorMap->getMin());
    list << QString::number(m_colorMap->getMax());
    list << QString("RGBAX");
    list << QString::number(m_colorMap->getNumSteps());

    for (int i = 0; i < m_colorMap->getNumSteps(); i++)
    {
        float r, g, b, a, x;
        m_colorMap->getStep(i, &r, &g, &b, &a, &x);

        list << QString::number(r);
        list << QString::number(g);
        list << QString::number(b);
        list << QString::number(a);
        list << QString::number(x);
    }

    QString tmp = list.join(" ");
    return tmp;
}

//!-------------------------------------------------------------------------
//! make the COLORMAP layout
//!-------------------------------------------------------------------------
void MEColorMapPort::makeLayout(layoutType type, QWidget *container)
{

    //create a vertical layout for 2 rows
    QVBoxLayout *vb = new QVBoxLayout(container);
    vb->setMargin(1);
    vb->setSpacing(1);

    // create first container widgets
    QWidget *w1 = new QWidget(container);
    vb->addWidget(w1);

    // create for each widget a horizontal layout
    QHBoxLayout *controlBox = new QHBoxLayout(w1);
    controlBox->setMargin(2);
    controlBox->setSpacing(2);

    // add filder pixmap
    folderAction[type] = new QPushButton();
    folderAction[type]->setFlat(true);
    folderAction[type]->setFocusPolicy(Qt::NoFocus);
    connect(folderAction[type], SIGNAL(clicked()), this, SLOT(folderCB()));
    if (m_fileOpen)
        folderAction[type]->setIcon(MEMainHandler::instance()->pm_folderopen);
    else
        folderAction[type]->setIcon(MEMainHandler::instance()->pm_folderclosed);
    controlBox->addWidget(folderAction[type]);

    // add combobox with predefined colormaps & m_preview
    if (type == MODULE)
        m_preview[type] = m_colorMap->getModulePreview();
    else
        m_preview[type] = m_colorMap->getControlPreview();

    m_preview[type]->show();
    controlBox->addWidget(m_preview[type], 1);

    if (MEMainHandler::instance()->cfg_TopLevelBrowser)
        manageDialog();

    else
    {
        // create second widget and layout for browser
        m_extendedPart[type] = new MEExtendedPart(container, this);
        vb->addWidget(m_extendedPart[type]);

        if (mapped && folderAction[MODULE])
            folderAction[MODULE]->setEnabled(false);
    }
}

//!-------------------------------------------------------------------------
//! create a colormap dialog widget (not embedded mode)
//!-------------------------------------------------------------------------
void MEColorMapPort::manageDialog()
{
    // create a dialog widget
    m_dialog = new QDialog(MEUserInterface::instance());
    QVBoxLayout *vbox = new QVBoxLayout(m_dialog);
    m_dialog->setWindowIcon(MEMainHandler::instance()->pm_logo);
    QString title = node->getTitle() + ":" + portname;
    m_dialog->setWindowTitle(MEMainHandler::instance()->generateTitle(title));
    connect(m_dialog, SIGNAL(finished(int)), this, SLOT(folderCB()));

    // add the color map widget
    m_colorMap->setParent(m_dialog);
    vbox->addWidget(m_colorMap, 1);

    // create the dialog buttons
    QDialogButtonBox *bb = new QDialogButtonBox();
    bb->addButton("Apply", QDialogButtonBox::AcceptRole);
    bb->addButton("Close", QDialogButtonBox::RejectRole);
    connect(bb, SIGNAL(accepted()), m_colorMap, SLOT(applyCB()));
    connect(bb, SIGNAL(rejected()), this, SLOT(colorMapClosed()));
    vbox->addWidget(bb);

    // user can close the window pressing ESC
    QAction *m_escape_a = new QAction("Escape", this);
    m_escape_a->setShortcut(Qt::Key_Escape);
    connect(m_escape_a, SIGNAL(triggered()), this, SLOT(colorMapClosed()));
    m_dialog->addAction(m_escape_a);
}

//!-------------------------------------------------------------------------
//! open/close the colormap
//!-------------------------------------------------------------------------
void MEColorMapPort::folderCB()
{
    m_fileOpen = !m_fileOpen;
    changeFolderPixmap();
    switchExtendedPart();
}

//!-------------------------------------------------------------------------
//! colormap was closed by user
//!-------------------------------------------------------------------------
void MEColorMapPort::colorMapClosed()
{
    m_fileOpen = false;
    changeFolderPixmap();
    switchExtendedPart();
}

//!-------------------------------------------------------------------------
//! this routine is always called when the user clicks the folder pixmap
//!-------------------------------------------------------------------------
void MEColorMapPort::changeFolderPixmap()
{

    // disable pixmap when mapped && embedded window exist
    if (mapped && !MEMainHandler::instance()->cfg_TopLevelBrowser)
    {
        if (folderAction[MODULE])
            folderAction[MODULE]->setEnabled(false);
    }

    else
    {
        if (folderAction[MODULE])
            folderAction[MODULE]->setEnabled(true);
    }

    // change pixmap
    if (!m_fileOpen)
    {
        if (folderAction[MODULE])
            folderAction[MODULE]->setIcon(MEMainHandler::instance()->pm_folderclosed);
        if (folderAction[CONTROL])
            folderAction[CONTROL]->setIcon(MEMainHandler::instance()->pm_folderclosed);
    }

    else
    {
        if (folderAction[MODULE])
            folderAction[MODULE]->setIcon(MEMainHandler::instance()->pm_folderopen);
        if (folderAction[CONTROL])
            folderAction[CONTROL]->setIcon(MEMainHandler::instance()->pm_folderopen);
    }
}

//!-------------------------------------------------------------------------
//! open/close the browser or colormap
//!-------------------------------------------------------------------------
void MEColorMapPort::switchExtendedPart()
{
    if (m_fileOpen)
    {
        if (m_dialog)
            m_dialog->show();

        else
        {
            if (mapped)
            {
                if (m_extendedPart[CONTROL])
                    m_extendedPart[CONTROL]->addColorMap();
            }

            else
            {
                if (m_extendedPart[MODULE])
                    m_extendedPart[MODULE]->addColorMap();
            }
        }
    }

    // close extended part or top level widget
    else
    {
        if (m_dialog)
            m_dialog->hide();

        else
        {
            if (mapped)
            {
                if (m_extendedPart[CONTROL])
                    m_extendedPart[CONTROL]->hide();
            }

            else
            {
                if (m_extendedPart[MODULE])
                    m_extendedPart[MODULE]->hide();
            }
        }
    }
}

//!-------------------------------------------------------------------------
//! map parameter to control panel
//!-------------------------------------------------------------------------
void MEColorMapPort::addToControlPanel()
{

    colorMapClosed();

    // create a module parameter window for the node
    if (!node->getControlInfo())
        node->createControlPanelInfo();

    // remove colormap from extended module prt
    if (m_colorMap && moduleLine)
    {
        m_fileOpen = false;
        if (m_extendedPart[MODULE])
            m_extendedPart[MODULE]->removeColorMap();
    }

    // create a control parameter line for this port
    if (controlLine == NULL)
    {
        QWidget *w = node->getControlInfo()->getContainer();
        controlLine = new MEControlParameterLine(w, this);
    }

    node->getControlInfo()->insertParameter(controlLine);
}

//!-------------------------------------------------------------------------
//! unmap parameter from control panel
//!-------------------------------------------------------------------------
void MEColorMapPort::removeFromControlPanel()
{
    colorMapClosed();

    if (controlLine != NULL)
    {
        if (m_colorMap)
        {
            m_fileOpen = false;
            if (m_extendedPart[CONTROL])
                m_extendedPart[CONTROL]->removeColorMap();
        }

        // remove comboBox & m_preview from layout
        // reparent
        QWidget *w = m_preview[CONTROL]->parentWidget();
        QLayout *lo = w->layout();
        lo->removeWidget(m_preview[CONTROL]);
        m_preview[CONTROL]->setParent(0);

        // remove contol line
        node->getControlInfo()->removeParameter(controlLine);

        // reset values
        m_extendedPart[CONTROL] = NULL;
        folderAction[CONTROL] = NULL;
        controlLine = NULL;
    }
}

#ifdef YAC
void MEColorMapPort::setValues(covise::coRecvBuffer &)
{
}
#endif
