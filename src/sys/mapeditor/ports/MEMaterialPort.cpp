/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QPushButton>
#include <QVBoxLayout>
#include <QAction>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDebug>

#include "MEMaterialPort.h"
#include "MELineEdit.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"
#include "widgets/MEUserInterface.h"
#include "material/MEMaterialDisplay.h"
#include "material/MEMaterialChooser.h"
#include "nodes/MENode.h"
#include "controlPanel/MEControlParameter.h"
#include "controlPanel/MEControlParameterLine.h"

/*!
    \class MEMaterialPort
    \brief handles module parameter of type material
*/

//!-------------------------------------------------------------------------
//!
//!-------------------------------------------------------------------------
MEMaterialPort::MEMaterialPort(MENode *node, QGraphicsScene *scene,
                               const QString &portname,
                               const QString &paramtype,
                               const QString &description)
    : MEParameterPort(node, scene, portname, paramtype, description)
    , m_fileOpen(false)
    , m_chooser(NULL)
{
    m_preview[MODULE] = m_preview[CONTROL] = NULL;
    folderAction[0] = folderAction[1] = NULL;
    m_dialog = NULL;
}

MEMaterialPort::MEMaterialPort(MENode *node, QGraphicsScene *scene,
                               const QString &portname,
                               int paramtype,
                               const QString &description,
                               int porttype)
    : MEParameterPort(node, scene, portname, paramtype, description, porttype)
    , m_fileOpen(false)
    , m_chooser(NULL)
{
    m_preview[MODULE] = m_preview[CONTROL] = NULL;
    folderAction[0] = folderAction[1] = NULL;
    m_dialog = NULL;
}

MEMaterialPort::~MEMaterialPort()
{
    if (m_dialog)
        delete m_dialog;
}

void MEMaterialPort::restoreParam()
{
    m_name = m_nameOld;
    m_values = m_valuesOld;
    sendParamMessage();
}

void MEMaterialPort::storeParam()
{
    m_nameOld = m_name;
    m_valuesOld = m_values;
}

//!-------------------------------------------------------------------------
//! module has requested parameter
//!-------------------------------------------------------------------------
void MEMaterialPort::moduleParameterRequest()
{
    sendParamMessage();
}

//!-------------------------------------------------------------------------
//! define one parameter
//!-------------------------------------------------------------------------
void MEMaterialPort::defineParam(QString value, int apptype)
{

#ifdef YAC

    Q_UNUSED(value);
    Q_UNUSED(apptype);

#else

    // define a material
    QStringList list = value.split(' ', QString::SkipEmptyParts);

    m_name = list[0];
    for (int j = 1; j < list.size(); j++)
        m_values.append(list[j].toFloat());

    MEParameterPort::defineParam(value, apptype);
#endif
}

//!-------------------------------------------------------------------------
//! modify one parameter
//!-------------------------------------------------------------------------
void MEMaterialPort::modifyParam(QStringList list, int noOfValues, int istart)
{
    Q_UNUSED(list);
    Q_UNUSED(noOfValues);
    Q_UNUSED(istart);
}

//!-------------------------------------------------------------------------
//! modify one parameter
//!-------------------------------------------------------------------------
void MEMaterialPort::modifyParameter(QString lvalue)
{
#ifdef YAC

    Q_UNUSED(lvalue);

#else

    QStringList list = lvalue.split(' ', QString::SkipEmptyParts);
    m_name = list[0];

    m_values.clear();
    for (int j = 1; j < list.size(); j++)
        m_values.append(list[j].toFloat());

    showMaterial();
    if (m_chooser)
        m_chooser->setMaterial(m_values);

#endif
}

void MEMaterialPort::sendParamMessage()
//!-------------------------------------------------------------------------
//! send a PARAM message to controller
//!-------------------------------------------------------------------------
{
    QStringList list;
    list << "ModuleDefined";

    for (int k = 0; k < m_values.size(); k++)
        list << QString::number(m_values[k]);

    QString tmp = list.join(" ");
    MEParameterPort::sendParamMessage(tmp);
}

//!-------------------------------------------------------------------------
//! make the layout
//!-------------------------------------------------------------------------
void MEMaterialPort::makeLayout(layoutType type, QWidget *container)
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

    // add folder pixmap
    folderAction[type] = new QPushButton();
    folderAction[type]->setFlat(true);
    folderAction[type]->setFocusPolicy(Qt::NoFocus);
    folderAction[type]->setToolTip("Open folder to edit the material");
    connect(folderAction[type], SIGNAL(clicked()), this, SLOT(folderCB()));
    if (m_fileOpen)
        folderAction[type]->setIcon(MEMainHandler::instance()->pm_folderopen);
    else
        folderAction[type]->setIcon(MEMainHandler::instance()->pm_folderclosed);
    controlBox->addWidget(folderAction[type]);

    // add a wigdet showing the current material
    if (!m_preview[type])
    {
        m_preview[type] = new MEMaterialDisplay(container);
        m_preview[type]->setValues(m_values);
    }
    m_preview[type]->show();
    controlBox->addWidget(m_preview[type]);
    controlBox->addStretch(10);

    manageDialog();
    if (m_chooser)
        m_chooser->setMaterial(m_values);
}

//!-------------------------------------------------------------------------
//! create a material dialog widget (not embedded mode)
//!-------------------------------------------------------------------------
void MEMaterialPort::manageDialog()
{
    m_dialog = new QDialog(0);
    m_dialog->setWindowIcon(MEMainHandler::instance()->pm_logo);
    QString title = node->getTitle() + ":" + portname;
    m_dialog->setWindowTitle(MEMainHandler::instance()->generateTitle(title));
    connect(m_dialog, SIGNAL(finished(int)), this, SLOT(folderCB()));

    m_chooser = new MEMaterialChooser();
    connect(m_chooser, SIGNAL(materialChanged(const QVector<float> &)), this, SLOT(materialChanged(const QVector<float> &)));
    m_chooser->setParent(m_dialog);

    QVBoxLayout *vbox = new QVBoxLayout(m_dialog);
    vbox->addWidget(m_chooser, 1);

    // create the dialog buttons
    QDialogButtonBox *bb = new QDialogButtonBox();
    QPushButton *pb = NULL;
    pb = bb->addButton("Apply", QDialogButtonBox::AcceptRole);
    pb = bb->addButton("Close", QDialogButtonBox::RejectRole);
    connect(bb, SIGNAL(accepted()), this, SLOT(applyCB()));
    connect(bb, SIGNAL(rejected()), this, SLOT(materialMapClosed()));
    pb = bb->addButton(QDialogButtonBox::Reset);
    connect(pb, SIGNAL(clicked()), this, SLOT(resetCB()));
    vbox->addWidget(bb);

    // user can close the window pressing ESC
    QAction *m_escape_a = new QAction("Escape", this);
    m_escape_a->setShortcut(Qt::Key_Escape);
    connect(m_escape_a, SIGNAL(triggered()), this, SLOT(materialMapClosed()));
    m_dialog->addAction(m_escape_a);
}

//------------------------------------------------------------------------
// unmap parameter from control panel
//------------------------------------------------------------------------
void MEMaterialPort::removeFromControlPanel()
{
    materialMapClosed();

    if (controlLine != NULL)
    {
        // remove comboBox & preview from layout
        // reparent
        QWidget *w = m_preview[CONTROL]->parentWidget();
        QLayout *lo = w->layout();
        lo->removeWidget(m_preview[CONTROL]);
        m_preview[CONTROL]->setParent(0);

        // remove control line
        node->getControlInfo()->removeParameter(controlLine);

        // reset values
        folderAction[CONTROL] = NULL;
        controlLine = NULL;
    }
}

//------------------------------------------------------------------------
// map parameter to control panel
//------------------------------------------------------------------------
void MEMaterialPort::addToControlPanel()
{
    materialMapClosed();

    // create a module parameter window for the node
    if (!node->getControlInfo())
        node->createControlPanelInfo();

    // remove colormap from extended module prt
    if (moduleLine)
        m_fileOpen = false;

    // create a control parameter line for this port
    if (controlLine == NULL)
    {
        QWidget *w = node->getControlInfo()->getContainer();
        controlLine = new MEControlParameterLine(w, this);
    }

    node->getControlInfo()->insertParameter(controlLine);
}

//------------------------------------------------------------------------
// open/close the colormap
//------------------------------------------------------------------------
void MEMaterialPort::folderCB()
{
    m_fileOpen = !m_fileOpen;
    changeFolderPixmap();
    if (m_fileOpen)
        storeParam();
}

//------------------------------------------------------------------------
// colormap was closed by user
//------------------------------------------------------------------------
void MEMaterialPort::materialMapClosed()
{
    m_fileOpen = false;
    changeFolderPixmap();
}

//------------------------------------------------------------------------
// this routine is always called when the user clicks the folder pixmap
//------------------------------------------------------------------------
void MEMaterialPort::changeFolderPixmap()
{

    // disable pixmap when mapped && embedded window exist
    if (folderAction[MODULE])
    {
        if (mapped)
            folderAction[MODULE]->setEnabled(false);

        else
            folderAction[MODULE]->setEnabled(true);
    }

    // change pixmap
    if (!m_fileOpen)
    {
        if (folderAction[MODULE])
            folderAction[MODULE]->setIcon(MEMainHandler::instance()->pm_folderclosed);
        if (folderAction[CONTROL])
            folderAction[CONTROL]->setIcon(MEMainHandler::instance()->pm_folderclosed);
        m_dialog->hide();
    }

    else
    {
        if (folderAction[MODULE])
            folderAction[MODULE]->setIcon(MEMainHandler::instance()->pm_folderopen);
        if (folderAction[CONTROL])
            folderAction[CONTROL]->setIcon(MEMainHandler::instance()->pm_folderopen);
        m_dialog->show();
    }
}

//------------------------------------------------------------------------
void MEMaterialPort::applyCB()
//------------------------------------------------------------------------
{
    showMaterial();
    sendParamMessage();
}

//------------------------------------------------------------------------
void MEMaterialPort::resetCB()
//------------------------------------------------------------------------
{
    restoreParam();
    if (m_chooser)
        m_chooser->setMaterial(m_values);
    showMaterial();
    sendParamMessage();
}

//------------------------------------------------------------------------
void MEMaterialPort::showMaterial()
//------------------------------------------------------------------------
{
    if (m_preview[MODULE])
        m_preview[MODULE]->setValues(m_values);

    if (m_preview[CONTROL])
        m_preview[CONTROL]->setValues(m_values);
}

//!
//! user has chosen a new material
//!
void MEMaterialPort::materialChanged(const QVector<float> &data)
{
    m_values.clear();
    m_values = data;
}

#ifdef YAC
void MEMaterialPort::setValues(covise::coRecvBuffer &)
{
}
#endif
