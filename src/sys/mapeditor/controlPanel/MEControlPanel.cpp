/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QVBoxLayout>

#include "MEControlPanel.h"
#include "MEControlParameter.h"

/*!
    \class MEControlPanel
    \brief Main control window for parameter ports
*/

MEControlPanel::MEControlPanel(QWidget *parent)
    : QFrame(parent)
//------------------------------------------------------------------------
{
    const char *text3 = "<p>The purpose of the <b>Control Panel</b> is the collection of graphical interactors "
                        "which typically represent often used and changed module parameters. </p>"
                        "<p> Widgets corresponding to the parameter types are used for the layout of interactor. "
                        "Interactors in the <b>Control Panel</b> allow the manipulation of parameters at every time "
                        "without the need to  pop up the <b>Module Parameter</b> window. "
                        "By clicking the toggle button in the <b>Module Parameter</b> window an interactor is generated. "
                        "Its representation then appears in the <b>Control Panel</b>. </p>";

    // create a main widget & layout
    m_boxLayout = new QVBoxLayout(this);
    m_boxLayout->setMargin(2);
    m_boxLayout->setSpacing(2);
    m_boxLayout->addStretch(1);

    setWhatsThis(text3);
    hide();
}

MEControlPanel *MEControlPanel::instance()
//------------------------------------------------------------------------
{
    static MEControlPanel *singleton = 0;
    if (singleton == 0)
        singleton = new MEControlPanel();

    return singleton;
}

MEControlPanel::~MEControlPanel()
//------------------------------------------------------------------------
{
    qDeleteAll(m_controlList);
    m_controlList.clear();
}

//!
//! add a new control information widget to the control panel
//!
void MEControlPanel::addControlInfo(MEControlParameter *info)
{
    m_boxLayout->insertWidget(m_controlList.size(), info);
    m_controlList.append(info);
}

//!
//! remove the control information widget from the control panel
//!
void MEControlPanel::removeControlInfo(MEControlParameter *info)
//------------------------------------------------------------------------
{
    m_controlList.remove(m_controlList.indexOf(info));
}

//!
//! switch the master/slave state
//!
void MEControlPanel::setMasterState(bool state)
//------------------------------------------------------------------------
{
    foreach (MEControlParameter *info, m_controlList)
        info->setMasterState(state);
}

//!
//! set a new proper size for the control content
//!
QSize MEControlPanel::sizeHint() const
{
    return QSize(200, 0);
}
