/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include <QPixmap>
#include <QLabel>

#include "TUILabel.h"
#include "TUIApplication.h"

/// Constructor
TUILabel::TUILabel(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    label = name;

    l = new QLabel(w);
    if (name.contains("."))
    {
        QPixmap pm(name);
        if (pm.isNull())
        {
            QString covisedir = QString(getenv("COVISEDIR"));
            QPixmap pm(covisedir + "/" + name);
            if (pm.isNull())
            {
                l->setText(label);
            }
            else
            {
                l->setPixmap(pm);
            }
        }
        else
        {
            l->setPixmap(pm);
        }
    }
    else
        l->setText(label);

    l->setMinimumSize(l->sizeHint());
    widget = l;
    setColor(Qt::black);
}

void TUILabel::setPixmap(const QPixmap &pm)
{
    l->resize(pm.size());
    l->setPixmap(pm);
}

/// Destructor
TUILabel::~TUILabel()
{
    delete widget;
    widget = NULL;
}

const char *TUILabel::getClassName() const
{
    return "TUILabel";
}

void TUILabel::setLabel(QString la)
{
    TUIElement::setLabel(la);
    //const char *dot = strchr(label,'.');
    //int len = strlen(label);
    //if(dot && dot < label+len && !(dot[1]>='0' && dot[1]<='9'))
    if (la.contains("."))
    {
        QPixmap pm(la);
        if (pm.isNull())
        {
            QString covisedir = QString(getenv("COVISEDIR"));
            QPixmap pm(covisedir + "/" + la);
            if (pm.isNull())
            {
                l->setText(la);
            }
            else
            {
                l->setPixmap(pm);
            }
        }
        else
        {
            l->setPixmap(pm);
        }
    }
    else
        l->setText(la);
}
