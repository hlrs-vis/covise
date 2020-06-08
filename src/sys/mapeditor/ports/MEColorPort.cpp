/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QHBoxLayout>
#include <QAction>
#include <QDialog>
#include <QDialogButtonBox>
#include <QtDebug>

#include "MEColorPort.h"
#include "MELineEdit.h"
#include "MEMessageHandler.h"
#include "widgets/MEUserInterface.h"
#include "handler/MEMainHandler.h"
#include "color/MEColorChooser.h"
#include "color/MEColorSelector.h"
#include "nodes/MENode.h"
#include "controlPanel/MEControlParameterLine.h"
#include "controlPanel/MEControlParameter.h"

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
MEColorPort::MEColorPort(MENode *node, QGraphicsScene *scene,
                         const QString &portname,
                         const QString &paramtype,
                         const QString &description)
    : MEParameterPort(node, scene, portname, paramtype, description)
    , m_fileOpen(false)
{
    folderAction[0] = folderAction[1] = NULL;
    m_preview[0] = m_preview[1] = NULL;
    manageDialog();
}

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
MEColorPort::MEColorPort(MENode *node, QGraphicsScene *scene,
                         const QString &portname,
                         int paramtype,
                         const QString &description,
                         int porttype)
    : MEParameterPort(node, scene, portname, paramtype, description, porttype)
    , m_fileOpen(false)
{
    folderAction[0] = folderAction[1] = NULL;
    m_preview[0] = m_preview[1] = NULL;
    m_chooser = new MEColorChooser();
    m_chooser->hide();
    manageDialog();
}

//------------------------------------------------------------------------
MEColorPort::~MEColorPort()
//------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------
// restore saved parameters
// after the user has pressed cancel in module parameter window
//------------------------------------------------------------------------
void MEColorPort::restoreParam()
{
    m_red = m_redold;
    m_green = m_greenold;
    m_blue = m_blueold;
    m_alpha = m_alphaold;
    sendParamMessage();
}

//------------------------------------------------------------------------
//save current value for further use
//------------------------------------------------------------------------
void MEColorPort::storeParam()
{
    m_redold = m_red;
    m_greenold = m_green;
    m_blueold = m_blue;
    m_alphaold = m_alpha;
}

//------------------------------------------------------------------------
// module has requested parameter
//------------------------------------------------------------------------
void MEColorPort::moduleParameterRequest()
{
    sendParamMessage();
}

//------------------------------------------------------------------------
// define one parameter of a module,	init from controller
//------------------------------------------------------------------------
void MEColorPort::defineParam(QString value, int apptype)
{
#ifdef YAC

    Q_UNUSED(value);
    Q_UNUSED(apptype);

#else

    QStringList list = value.split(" ", QString::SkipEmptyParts);
    if (list.count() != 4)
    {
        QString msg = "MEParameterPort::modifyParam: " + node->getNodeTitle() + ": Parameter type " + parameterType + " has wrong number of values";
        MEUserInterface::instance()->printMessage(msg);
    }

    else
    {
        m_red = list[0].toFloat();
        m_green = list[1].toFloat();
        m_blue = list[2].toFloat();
        m_alpha = list[3].toFloat();

        MEParameterPort::defineParam(value, apptype);

        updateItems();
    }

    m_fileOpen = false;

#endif
}

//------------------------------------------------------------------------
// modify one parameter
//------------------------------------------------------------------------
void MEColorPort::modifyParam(QStringList, int, int)
{
}

//------------------------------------------------------------------------
// modify one parameter of a module,	update param  from controller
//------------------------------------------------------------------------
void MEColorPort::modifyParameter(QString lvalue)
{
#ifdef YAC

    Q_UNUSED(lvalue);

#else

    lvalue = lvalue.trimmed();

    QStringList list = QString(lvalue).split(" ");

    if (list.count() == 4)
    {
        m_red = list[0].toFloat();
        m_green = list[1].toFloat();
        m_blue = list[2].toFloat();
        m_alpha = list[3].toFloat();

        updateItems();
    }

    else
    {
        QString msg = "MEParameterPort::modifyParam: " + node->getNodeTitle() + ": Parameter type " + parameterType + " has wrong number of values";
        MEUserInterface::instance()->printMessage(msg);
    }
#endif
}

//------------------------------------------------------------------------
// update all widgets in module parameter window & control panel
//------------------------------------------------------------------------
void MEColorPort::updateItems()
{
    m_color.setRgb((int)(m_red * 255.), (int)(m_green * 255.), (int)(m_blue * 255.), (int)(m_alpha * 255.));

    // modify module & control line content
    if (!m_dataList[MODULE].isEmpty())
    {
        m_dataList[MODULE].at(0)->setText(QString().setNum(m_red));
        m_dataList[MODULE].at(1)->setText(QString().setNum(m_green));
        m_dataList[MODULE].at(2)->setText(QString().setNum(m_blue));
        m_dataList[MODULE].at(3)->setText(QString().setNum(m_alpha));
    }

    if (!m_dataList[CONTROL].isEmpty())
    {
        m_dataList[CONTROL].at(0)->setText(QString().setNum(m_red));
        m_dataList[CONTROL].at(1)->setText(QString().setNum(m_green));
        m_dataList[CONTROL].at(2)->setText(QString().setNum(m_blue));
        m_dataList[CONTROL].at(3)->setText(QString().setNum(m_alpha));
    }

    if (m_preview[MODULE])
        m_preview[MODULE]->setPalette(QPalette(m_color));

    if (m_preview[CONTROL])
        m_preview[CONTROL]->setPalette(QPalette(m_color));

    if (m_dialog)
        m_chooser->setColor(m_color);

    m_currColor = m_color;
    m_currAlpha = m_color.alpha();
}

#define addColor(text)                                                                         \
    le = new MELineEdit();                                                                     \
    le->setMinimumWidth(100);                                                                  \
    le->setText(QString::number(text));                                                        \
    connect(le, SIGNAL(contentChanged(const QString &)), this, SLOT(textCB(const QString &))); \
    connect(le, SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));                     \
    m_dataList[type].append(le);                                                               \
    controlBox->addWidget(le, 1);

//!
//! decide wich layout should be created
//!
void MEColorPort::makeLayout(layoutType type, QWidget *container)
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

    // add folder pixmap
    folderAction[type] = new QPushButton();
    folderAction[type]->setFlat(true);
    folderAction[type]->setFocusPolicy(Qt::NoFocus);
    connect(folderAction[type], SIGNAL(clicked()), this, SLOT(folderCB()));
    if (m_fileOpen)
        folderAction[type]->setIcon(MEMainHandler::instance()->pm_folderopen);
    else
        folderAction[type]->setIcon(MEMainHandler::instance()->pm_folderclosed);
    controlBox->addWidget(folderAction[type]);

    // add colorpicker
    m_colorPicker[type] = new MEColorSelector();
    connect(m_colorPicker[type], SIGNAL(clicked()), m_colorPicker[type], SLOT(selectedColorCB()));
    connect(m_colorPicker[type], SIGNAL(pickedColor(const QColor &)), this, SLOT(showColor(const QColor &)));
    controlBox->addWidget(m_colorPicker[type]);

    // add combobox with predefined colormaps & preview
    if (!m_preview[type])
    {
        m_preview[type] = new MEColorDisplay(container);
        m_preview[type]->showColor(m_color);
    }
    m_preview[type]->show();

    controlBox->addWidget(m_preview[type], 1);

    MELineEdit *le;
    //QRegExp rx("[0-1][.]\\d{0,6}");
    //QRegExpValidator *validator = new QRegExpValidator(rx, this);
    addColor(m_red);
    addColor(m_green);
    addColor(m_blue);
    addColor(m_alpha);
}

void MEColorPort::sendParamMessage()
//------------------------------------------------------------------------
/* send a PARAM message to controller				                        */
/* key	    ______	    keyword for message			                  	*/
//------------------------------------------------------------------------
{
    QString buffer;
    buffer = QString::number(m_red) + " " + QString::number(m_green) + " ";
    buffer = buffer + QString::number(m_blue) + " " + QString::number(m_alpha);

    MEParameterPort::sendParamMessage(buffer);
}

//!-------------------------------------------------------------------------
//! create a colormap dialog widget (not embedded mode)
//!-------------------------------------------------------------------------
void MEColorPort::manageDialog()
{
    m_dialog = new QDialog(0);
    m_dialog->setWindowIcon(MEMainHandler::instance()->pm_logo);
    QString title = node->getTitle() + ":" + portname;
    m_dialog->setWindowTitle(MEMainHandler::instance()->generateTitle(title));
    connect(m_dialog, SIGNAL(finished(int)), this, SLOT(folderCB()));

    m_chooser = new MEColorChooser();
    connect(m_chooser, SIGNAL(colorChanged(const QColor &)), this, SLOT(newColor(const QColor &)));
    m_chooser->setParent(m_dialog);

    QVBoxLayout *vbox = new QVBoxLayout(m_dialog);
    vbox->addWidget(m_chooser, 1);

    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(bb, SIGNAL(rejected()), this, SLOT(folderCB()));
    connect(bb, SIGNAL(accepted()), this, SLOT(okCB()));
    connect(bb, SIGNAL(accepted()), this, SLOT(folderCB()));
    vbox->addWidget(bb);

    // user can close the window pressing ESC
    QAction *m_escape_a = new QAction("Escape", this);
    m_escape_a->setShortcut(Qt::Key_Escape);
    connect(m_escape_a, SIGNAL(triggered()), this, SLOT(folderCB()));
    m_dialog->addAction(m_escape_a);
}

//------------------------------------------------------------------------
// get the float vector value
//------------------------------------------------------------------------
void MEColorPort::textCB(const QString &)
{

    // object that sent the signal

    const QObject *obj = sender();
    MELineEdit *le = (MELineEdit *)obj;

    // find widget that send the  in list

    layoutType type = MODULE;
    if (m_dataList[MODULE].contains(le))
        type = MODULE;

    else if (m_dataList[CONTROL].contains(le))
        type = CONTROL;

    else
        qCritical() << "did not find Color line edit" << endl;

#ifdef YAC

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;
    sb << 4 << m_dataList[type].at(0)->text().toFloat();
    sb << m_dataList[type].at(1)->text().toFloat();
    sb << m_dataList[type].at(2)->text().toFloat();
    sb << m_dataList[type].at(3)->text().toFloat();

    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    m_red = m_dataList[type].at(0)->text().toFloat();
    m_green = m_dataList[type].at(1)->text().toFloat();
    m_blue = m_dataList[type].at(2)->text().toFloat();
    m_alpha = m_dataList[type].at(3)->text().toFloat();
    sendParamMessage();
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

//------------------------------------------------------------------------
// unmap parameter from control panel
//------------------------------------------------------------------------
void MEColorPort::removeFromControlPanel()
{
    colorMapClosed();

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
void MEColorPort::addToControlPanel()
{
    colorMapClosed();

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
void MEColorPort::folderCB()
{
    m_fileOpen = !m_fileOpen;
    changeFolderPixmap();
}

//------------------------------------------------------------------------
// colormap was closed by user
//------------------------------------------------------------------------
void MEColorPort::colorMapClosed()
{
    m_fileOpen = false;
    changeFolderPixmap();
}

//------------------------------------------------------------------------
// this routine is always called when the user clicks the folder pixmap
//------------------------------------------------------------------------
void MEColorPort::changeFolderPixmap()
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
void MEColorPort::okCB()
//------------------------------------------------------------------------
{
    // set chosen color & show it
    QColor color(m_currColor);
    color.setAlpha(m_currAlpha);
    showColor(color);
}

//------------------------------------------------------------------------
void MEColorPort::showColor(const QColor &color)
//------------------------------------------------------------------------
{
    m_red = color.redF();
    m_green = color.greenF();
    m_blue = color.blueF();
    m_alpha = color.alphaF();

    if (m_preview[MODULE])
        m_preview[MODULE]->showColor(color);

    if (m_preview[CONTROL])
        m_preview[CONTROL]->showColor(color);

    // send color to controller
    update();
    sendParamMessage();
}

//!
//! user has chosen a new  color
//!
void MEColorPort::newColor(const QColor &col)
{
    m_currColor = col;
}

#ifdef YAC

//------------------------------------------------------------------------
void MEColorPort::setValues(covise::coRecvBuffer &tb)
//------------------------------------------------------------------------
{
    tb >> m_red;
    tb >> m_green;
    tb >> m_blue;
    tb >> m_alpha;

    updateItems();
}
#endif
