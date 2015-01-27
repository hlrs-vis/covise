/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef YAC
#include "XPM/yac.xpm"
#else
#include "XPM/covise.xpm"
#endif

#include <qlayout.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include <QGridLayout>
#include <QPixmap>
#include <QHBoxLayout>

#include "InvClipPlaneEditor.h"
#include "InvViewer.h"
#ifndef YAC
#include "InvMain.h"
#else
#include "InvMain_yac.h"
#endif

//=========================================================================
//
InvClipPlaneEditor::InvClipPlaneEditor(QWidget *parent, const char *name)
    : QDialog(parent)
//
//=========================================================================
{

    setWindowTitle("Clipping Plane");

    // Default for point and normal
    pt[0] = 0.0;
    pt[1] = 0.0;
    pt[2] = 0.0;

    nl[0] = 1.0;
    nl[1] = 0.0;
    nl[2] = 0.0;

    // make the layout

    QGridLayout *fbox = new QGridLayout(this); // 2,4

    QLabel *label = new QLabel(this);
    label->setText("Point");
    label->setAlignment(Qt::AlignVCenter | Qt::AlignHCenter);
    fbox->addWidget(label, 0, 0);

    label = new QLabel(this);
    label->setText("Normal");
    label->setAlignment(Qt::AlignVCenter | Qt::AlignHCenter);
    fbox->addWidget(label, 1, 0);

    int i;
    for (i = 0; i < 3; i++)
    {
        p[i] = new QLineEdit(this);
        p[i]->setModified(true);
        p[i]->setText(QString().setNum(pt[i]));
        fbox->addWidget(p[i], 0, i + 1);
    }

    for (i = 0; i < 3; i++)
    {
        n[i] = new QLineEdit(this);
        n[i]->setModified(true);
        n[i]->setText(QString().setNum(nl[i]));
        fbox->addWidget(n[i], 1, i + 1);
    }

    QHBoxLayout *box = new QHBoxLayout(this); // 20
    QPushButton *ok = new QPushButton("Apply", this);
    box->addWidget(ok);

    connect(ok, SIGNAL(clicked()), this, SLOT(equationCB()));

    // set the logo
    setWindowIcon(QPixmap(logo));
}

//=========================================================================
//
//    Destructor.
//
//=========================================================================
InvClipPlaneEditor::~InvClipPlaneEditor()
{
}

//
// called whenever the "Apply" button is pushed
//
void InvClipPlaneEditor::equationCB()
{
    for (int i = 0; i < 3; i++)
    {
        pt[i] = atof(p[i]->text().toLatin1());
        nl[i] = atof(n[i]->text().toLatin1());
    }
    SbVec3f point(pt[0], pt[1], pt[2]);
    SbVec3f normal(nl[0], nl[1], nl[2]);

    renderer->viewer->setClipPlaneEquation(normal, point);
}
