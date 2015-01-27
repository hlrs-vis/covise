/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QLabel>
#include <QHBoxLayout>
#include <QPushButton>

#include "MESliderPort.h"
#include "MELineEdit.h"
#include "handler/MEMainHandler.h"

/*!
    \class MESliderPort
    \brief Base class for parameters of type SLIDER
*/

MESliderPort::MESliderPort(MENode *node, QGraphicsScene *scene,
                           const QString &portname,
                           const QString &paramtype,
                           const QString &description)
    : MEParameterPort(node, scene, portname, paramtype, description)
{
    m_textField = NULL;
    m_slider[MODULE] = m_slider[CONTROL] = NULL;
}

MESliderPort::MESliderPort(MENode *node, QGraphicsScene *scene,
                           const QString &portname,
                           int paramtype,
                           const QString &description,
                           int porttype)
    : MEParameterPort(node, scene, portname, paramtype, description, porttype)
{
    m_textField = NULL;
    m_slider[MODULE] = m_slider[CONTROL] = NULL;
}

MESliderPort::~MESliderPort()
{
}

//!
//! unmap parameter from control panel
//!
void MESliderPort::removeFromControlPanel()
{
    MEParameterPort::removeFromControlPanel();
    m_slider[CONTROL] = NULL;
    m_textField = NULL;
}

//!
//! make a control line widget for the control panel
//!
void MESliderPort::makeControlLine(layoutType type, QWidget *w, const QString &value)
{

    // create a vertical layout for 2 rows

    QHBoxLayout *vb = new QHBoxLayout(w);
    vb->setMargin(2);
    vb->setSpacing(2);

    // text editor line

    m_textField = new MELineEdit(w);
    m_textField->setText(value);
    m_textField->setMinimumWidth(MEMainHandler::instance()->getSliderWidth());
    vb->addWidget(m_textField, 2);

    connect(m_textField, SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));
    connect(m_textField, SIGNAL(contentChanged(const QString &)), this, SLOT(text2CB(const QString &)));

    // slider
    m_slider[type] = makeSlider(w);
    vb->addWidget(m_slider[type], 6);
}

#define addValue(text, value)                                                                  \
    l = new QLabel(text, secondLine);                                                          \
    l->setFixedWidth(45);                                                                      \
    hb2->addWidget(l);                                                                         \
    le = new MELineEdit(secondLine);                                                           \
    le->setMinimumWidth(MEMainHandler::instance()->getSliderWidth());                          \
    le->setText(value);                                                                        \
    connect(le, SIGNAL(contentChanged(const QString &)), this, SLOT(textCB(const QString &))); \
    connect(le, SIGNAL(editingFinished()), this, SLOT(boundaryCB()));                          \
    connect(le, SIGNAL(returnPressed()), this, SLOT(boundaryCB()));                            \
    hb2->addWidget(le, 4);                                                                     \
    m_editList << le;

//!
//! make a control line widget for the control panel
//!
void MESliderPort::makeModuleLine(layoutType type, QWidget *w, const QStringList &values)
{

    // create a vertical layout for 2 rows

    QVBoxLayout *vb = new QVBoxLayout(w);
    vb->setMargin(2);
    vb->setSpacing(2);

    // create two container widgets and layouts

    QWidget *firstLine = new QWidget(w);
    QHBoxLayout *hb1 = new QHBoxLayout(firstLine);
    hb1->setMargin(2);
    hb1->setSpacing(2);

    secondLine = new QWidget(w);
    QHBoxLayout *hb2 = new QHBoxLayout(secondLine);
    hb2->setMargin(2);
    hb2->setSpacing(2);
    secondLine->hide();

    vb->addWidget(firstLine);
    vb->addWidget(secondLine);

    // text editor line

    MELineEdit *le = new MELineEdit(firstLine);
    le->setText(values[2]);
    le->setMinimumWidth(MEMainHandler::instance()->getSliderWidth());
    connect(le, SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));
    connect(le, SIGNAL(contentChanged(const QString &)), this, SLOT(textCB(const QString &)));
    connect(le, SIGNAL(editingFinished()), this, SLOT(boundaryCB()));
    connect(le, SIGNAL(returnPressed()), this, SLOT(boundaryCB()));
    m_editList << le;
    hb1->addWidget(le, 2);

    // slider

    m_slider[type] = makeSlider(firstLine);
    hb1->addWidget(m_slider[type], 6);

    // make the second widget line (normally hidden)

    QLabel *l;
    addValue("Min ", values[0]);
    addValue("Max ", values[1]);
    addValue("Step", values[3]);
}

#define addButton(pb, pixmap, callback)                     \
    pb = new QPushButton();                                 \
    pb->setIcon(pixmap);                                    \
    connect(pb, SIGNAL(clicked()), this, SLOT(callback())); \
    connect(pb, SIGNAL(clicked()), pb, SLOT(setFocus()));   \
    hbox->addWidget(pb);

//!
//! create a player widget like recorder (Controlpanel)
//!
void MESliderPort::makePlayer(layoutType, QWidget *w, const QString &value)
{

    // horizontal Layout

    QHBoxLayout *hbox = new QHBoxLayout(w);
    hbox->setMargin(2);
    hbox->setSpacing(2);

    // text filled with current value

    m_textField = new MELineEdit(w);
    m_textField->setMinimumWidth(MEMainHandler::instance()->getSliderWidth());
    m_textField->setText(value);
    connect(m_textField, SIGNAL(contentChanged(const QString &)), this, SLOT(text2CB(const QString &)));
    connect(m_textField, SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));
    hbox->addWidget(m_textField, 2);

    // add player buttons

    addButton(left2, QPixmap(":/icons/2leftarrow.png"), reverseCB);
    addButton(left1, QPixmap(":/icons/1leftarrow.png"), left1CB);
    addButton(stopp, QPixmap(":/icons/playerstop.png"), stopCB);
    addButton(right1, QPixmap(":/icons/1rightarrow.png"), right1CB);
    addButton(right2, QPixmap(":/icons/2rightarrow.png"), playCB);
}
