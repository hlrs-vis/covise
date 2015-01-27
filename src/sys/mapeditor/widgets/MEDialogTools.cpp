/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <QDialogButtonBox>
#include <QDebug>
#include <QLabel>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QListWidget>
#include <QLineEdit>
#include <QDialogButtonBox>
#include <QMessageBox>

#if QT_VERSION >= 0x040400
#include <QFormLayout>
#endif

#include "MEDialogTools.h"
#include "MEGraphicsView.h"
#include "handler/MEMainHandler.h"
#include "handler/MENodeListHandler.h"
#include "handler/MEHostListHandler.h"

/*!
    \class MERenameDialog
    \brief Dialog box used for renaming of module nodes and groups
*/

MERenameDialog::MERenameDialog(int mode, const QString &text, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(MEMainHandler::instance()->generateTitle("Rename"));

    QVBoxLayout *box = new QVBoxLayout(this);

    if (mode == GROUP) // group
    {
        QLabel *label = new QLabel();
        label->setText("Add prefix to grouped nodes");
        label->setFont(MEMainHandler::s_boldFont);
        label->setAlignment(Qt::AlignHCenter);
        box->addWidget(label);

        m_renameLineEdit = new QLineEdit();
        box->addWidget(m_renameLineEdit);
        connect(m_renameLineEdit, SIGNAL(returnPressed()), this, SLOT(accepted()));
    }

    else // single
    {

#if QT_VERSION >= 0x040400

        m_renameLineEdit = new QLineEdit();
        connect(m_renameLineEdit, SIGNAL(returnPressed()), this, SLOT(accepted()));

        QFormLayout *grid = new QFormLayout();
        grid->addRow("Old title", new QLabel(text));
        grid->addRow("New title", m_renameLineEdit);
        box->addLayout(grid);

#else

        QGridLayout *grid = new QGridLayout();
        box->addLayout(grid);

        QLabel *label = new QLabel();
        label->setText("Old title   ");
        label->setFont(MEMainHandler::s_boldFont);
        grid->addWidget(label, 0, 0);

        m_renameLineEdit = new QLineEdit();
        m_renameLineEdit->setReadOnly(true);
        m_renameLineEdit->setText(text);
        grid->addWidget(m_renameLineEdit, 0, 1);

        label = new QLabel();
        label->setText("New title   ");
        label->setFont(MEMainHandler::s_boldFont);
        grid->addWidget(label, 1, 0);

        m_renameLineEdit = new QLineEdit();
        grid->addWidget(m_renameLineEdit, 1, 1);
        connect(m_renameLineEdit, SIGNAL(returnPressed()), this, SLOT(accepted()));

#endif
    }

    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(bb, SIGNAL(accepted()), this, SLOT(accepted()));
    connect(bb, SIGNAL(rejected()), this, SLOT(rejected()));
    box->addWidget(bb);
}

//!
//! accept a rename action
//!
void MERenameDialog::accepted()
{
    // get selected name
    QString newText = m_renameLineEdit->text();

    if (newText.length() == 0)
    {
        QMessageBox::information(0, windowTitle(),
                                 "Zero length labels are not allowed.\nPlease choose a different name!");
        hide();
        return;
    }

    // rename modules
    if (MENodeListHandler::instance()->nameAlreadyExist(newText))
    {
        QMessageBox::information(0, windowTitle(),
                                 "Label " + newText + " already exists in the network.\nPlease choose a different name!");

        hide();
        return;
    }

// send message
#ifndef YAC
    MEGraphicsView::instance()->renameNodes(newText);
#endif

    hide();
    MEMainHandler::instance()->mapWasChanged("RENAME");
}

//!
//! reject a host selection
//!
void MERenameDialog::rejected()
{
    hide();
}

/*!
    \class MEDeleteHostDialog
    \brief Dialog box used host deletion
*/
MEDeleteHostDialog::MEDeleteHostDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(MEMainHandler::instance()->generateTitle("Delete Host"));

    QVBoxLayout *box = new QVBoxLayout(this);

    // caption
    QLabel *label = new QLabel();
    label->setText("Current session partners");
    label->setAlignment(Qt::AlignCenter);
    label->setFont(MEMainHandler::instance()->s_boldFont);
    box->addWidget(label);

    // fill box with hosts
    m_deleteHostBox = new QListWidget();
    m_deleteHostBox->addItems(MEHostListHandler::instance()->getList2());
    m_deleteHostBox->setSelectionMode(QAbstractItemView::SingleSelection);
    box->addWidget(m_deleteHostBox, 1);

    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(bb, SIGNAL(accepted()), this, SLOT(accepted()));
    connect(bb, SIGNAL(rejected()), this, SLOT(rejected()));
    box->addWidget(bb);
}

//!
//! return the selected string
//!
QString MEDeleteHostDialog::getLine()
{
    return m_deleteHostBox->currentItem()->text();
}

//!
//! accept a host selection
//!
void MEDeleteHostDialog::accepted()
{
    qDebug() << m_deleteHostBox->currentItem()->text();
    hide();
    accept();
}

//!
//! reject a host selection
//!
void MEDeleteHostDialog::rejected()
{
    hide();
    reject();
}

/*!
    \class MEMirrorHostDialog
    \brief Dialog box used host deletion
*/
MEMirrorHostDialog::MEMirrorHostDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(MEMainHandler::instance()->generateTitle("Set Mirror Hosts"));

    /*
      // make the main layout
      // create a grid layout for module parameter information
      // disable stretching of first column

      QGridLayout *mlist = new QGridLayout(m_mirrorBox);
      mlist->setColumnStretch(0, 0);
      mlist->setColumnStretch(1, 1);

      // caption
      QLabel *label = new QLabel(m_mirrorBox);
      label->setText("Hosts");
      label->setAlignment(Qt::AlignCenter);
      label->setFont(s_boldFont);
      mlist->addWidget(label, 0, 0);

      label = new QLabel(m_mirrorBox);
      label->setText("List of mirrors");
      label->setAlignment(Qt::AlignCenter);
      label->setFont(s_boldFont);
      mlist->addWidget(label, 0, 1);

      QFrame *f = new QFrame(m_mirrorBox);
      f->setFrameStyle(QFrame::HLine | QFrame::Raised);
      mlist->addWidget(f, 1, 0);

      f = new QFrame(m_mirrorBox);
      f->setFrameStyle(QFrame::HLine | QFrame::Raised);
      mlist->addWidget(f, 1, 1);

      // copy hostlist
      foreach ( MEHost *nptr, hostList)
         m_syncList << nptr;

      // create table woth possible hosts for mirroring
      int it = 2;
      QCheckBox *rb;
      myButtonGroup *group;
      foreach ( MEHost *host, m_syncList )
      {
         // current host
         label = new QLabel(m_mirrorBox);
         label->setText(host->getShortname() + " (" + host->getIPAddress() +")");
         label->setAlignment(Qt::AlignCenter);
         QPalette palette;
         palette.setBrush(label->backgroundRole(), host->getColor());
         label->setPalette(palette);
         label->setFrameStyle(QFrame::Panel | QFrame::Raised);
         mlist->addWidget(label, it, 0);

         // list of available mirrors
         group = new myButtonGroup(host, m_mirrorBox);
         connect( group, SIGNAL(clicked(int)), this, SLOT(selectMirrorHost(int)) );
         mlist->addWidget(group, it, 1);
         it++;

         foreach ( MEHost *nptr, hostList)
         {
            if(host != nptr)
            {
               rb = new QCheckBox(group);
               rb->setText(nptr->getShortname() + " (" + nptr->getIPAddress() +")");
               QPalette palette;
               palette.setBrush(rb->backgroundRole(), nptr->getColor());
               rb->setPalette(palette);
               foreach ( MEHost *mirror, host->mirrorNames)
               {
                  if(mirror == nptr)
                     rb->setChecked(true);
                  else
                     rb->setChecked(false);
               }
            }
         }
      }

      m_syncList.clear();

      // add space & close button

      QPushButton *no = new QPushButton("Close", m_mirrorBox);
      connect(no, SIGNAL(clicked()), this, SLOT(rejected()));
      mlist->addWidget(no, it, it, 0, 1);
   */
}

//!
//! callback from mirror list, id = no of the QCheckBox
//!
void MEMirrorHostDialog::selectMirrorHost(int)
{
    /*
   // object that sent the signal
   const QObject *obj = sender();
   myButtonGroup *box = (myButtonGroup *) obj;

   // get original host
   MEHost *host = box->getMainHost();

   // look for name of mirror host
   QCheckBox *bt  = (QCheckBox *)box->find(id);
   QString name = bt->text().section("(", 1, 1);
   MEHost *h2 = getHostofNode2(name.section(")", 0, 0));

   // add  or remove mirror host to original host
   if(bt->isChecked() && !host->mirrorNames.contains(h2) )
      host->mirrorNames << h2;
   else
      host->mirrorNames << h2;*/
}
