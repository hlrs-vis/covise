/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <QComboBox>
#include <QLabel>
#include <QGridLayout>
#include <QCheckBox>

#ifdef YAC
#include "yac/coQTSendBuffer.h"
#endif

#include "MEPreference.h"
#include "MEUserInterface.h"
#include "MEMessageHandler.h"
#include "color/MEColorMap.h"

;

/*!
   \class MEPreference
   \brief This class provides a widget for displaying the preferences for YAC (obsolete)
*/

MEPreference::MEPreference(QWidget *parent)
    : QFrame(parent)

{

    // set style
    setFrameStyle(QFrame::Box | QFrame::Sunken);

    // create a new grid layout
    //----------------------------------------------------------------------
    int row = 0;
    QGridLayout *grid = new QGridLayout(this);

    QLabel *label = new QLabel("Data Caching Mode", this);
    caching = new QComboBox(this);
    caching->insertItem(0, "None");
    caching->insertItem(1, "Low");
    caching->insertItem(2, "High");
    connect(caching, SIGNAL(activated(int)), this, SLOT(cachSelected(int)));
    grid->addWidget(label, row, 0);
    grid->addWidget(caching, row, 1);
    row++;

    //----------------------------------------------------------------------
    label = new QLabel("RegistryMode", this);
    registry = new QComboBox(this);
    registry->insertItem(0, "None");
    registry->insertItem(1, "Global Only");
    registry->insertItem(2, "All");
    connect(registry, SIGNAL(activated(int)), this, SLOT(regModeSelected(int)));
    grid->addWidget(label, row, 0);
    grid->addWidget(registry, row, 1);
    row++;

    //----------------------------------------------------------------------
    label = new QLabel("ExecutionOnConnect", this);
    execConn = new QCheckBox(this);
    connect(execConn, SIGNAL(clicked()), this, SLOT(execConnSelected()));
    grid->addWidget(label, row, 0);
    grid->addWidget(execConn, row, 1);
    row++;

    //----------------------------------------------------------------------
    label = new QLabel("DebugMode", this);
    debugToggle = new QCheckBox(this);
    connect(debugToggle, SIGNAL(clicked()), this, SLOT(debugPressed()));
    grid->addWidget(label, row, 0);
    grid->addWidget(debugToggle, row, 1);
    row++;

    grid->setColumnStretch(2, 10);
    grid->setRowStretch(row, 10);
}

//------------------------------------------------------------------------
MEPreference *MEPreference::instance()
//------------------------------------------------------------------------
{
    static MEPreference *singleton = 0;
    if (singleton == 0)
        singleton = new MEPreference();

    return singleton;
}

MEPreference::~MEPreference()
{
}

//!
//! Updating of incoming changes
//!
void MEPreference::update(QString name, QString value)
{

    // data caching
    if (name == "DataCaching")
        caching->setCurrentIndex(caching->findText(value));

    // registry modes
    else if (name == "RegistryMode")
        registry->setCurrentIndex(caching->findText(value));

    // debug mode
    else if (name == "DebugMode")
    {
        if (value == "true")
            debugToggle->setChecked(true);
        else
            debugToggle->setChecked(false);
    }

    // default colormap
    else if (name == "CurrentCMap")
        MEUserInterface::instance()->getColorMap()->setPredefinedMap(value);

    // execution models
    else if (name == "ExecutionOnConnect")
    {
        if (value == "true")
            execConn->setChecked(true);
        else
            execConn->setChecked(false);
    }
}

void MEPreference::cachSelected(int id)
{
#ifdef YAC
    covise::coSendBuffer sb;
    sb << "UI" << 0 << "DataCaching";
    QString s = caching->itemText(id);
    sb << s.toAscii().data();
    MEMessageHandler::instance()->sendControlMessage(covise::coCtrlMsg::REGISTRY_SET_VALUE, sb);
#else
    Q_UNUSED(id);
#endif
}

void MEPreference::regModeSelected(int id)

{
#ifdef YAC
    covise::coSendBuffer sb;
    sb << "UI" << 0 << "RegistryMode";
    QString s = registry->itemText(id);
    sb << s.toAscii().data();
    MEMessageHandler::instance()->sendControlMessage(covise::coCtrlMsg::REGISTRY_SET_VALUE, sb);
#else
    Q_UNUSED(id);
#endif
}

void MEPreference::debugPressed()
{
#ifdef YAC
    covise::coSendBuffer sb;
    sb << "UI" << 0 << "DebugMode";
    if (debugToggle->isChecked())
        sb << "true";

    else
        sb << "false";
    MEMessageHandler::instance()->sendControlMessage(covise::coCtrlMsg::REGISTRY_SET_VALUE, sb);
#endif
}

void MEPreference::execConnSelected()
{
#ifdef YAC
    covise::coSendBuffer sb;
    sb << "UI" << 0 << "ExecutionOnConnect";
    if (execConn->isChecked())
        sb << "true";
    else
        sb << "false";
    MEMessageHandler::instance()->sendControlMessage(covise::coCtrlMsg::REGISTRY_SET_VALUE, sb);
#endif
}
