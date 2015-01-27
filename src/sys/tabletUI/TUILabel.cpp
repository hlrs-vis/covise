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
                l->setText(name);
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
        l->setText(name);

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

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUILabel::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUILabel::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

void TUILabel::setColor(Qt::GlobalColor color)
{
    TUIElement::setColor(color);
}

char *TUILabel::getClassName()
{
    return (char *)"TUILabel";
}

bool TUILabel::isOfClassName(char *classname)
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return TUIElement::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

void TUILabel::setValue(int type, covise::TokenBuffer &tb)
{
    //cerr << "setValue " << type << endl;
    TUIElement::setValue(type, tb);
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
