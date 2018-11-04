/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <string.h>

#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#include <config/CoviseConfig.h>
#else
#include <wce_msg.h>
#endif
#include "TUIElement.h"
#include "TUIContainer.h"
#include "TUIApplication.h"

/// Constructor.
TUIElement::TUIElement(int id, int /*type*/, QWidget * /*w*/, int parent, QString name)
{
    widget = NULL;
    parentContainer = NULL;
    enabled = true;
    hidden = false;
    visible = false;
    highlighted = false;
    ParentID = parent;
    ID = id;
    label = "";
    xPos = 0;
    yPos = 0;
    width = 1;
    height = 1;
    this->name = name;
    TUIMainWindow::getInstance()->addElement(this);
}

/// Destructor
TUIElement::~TUIElement()
{
    widget = nullptr;
    widgets.clear();

    if (parentContainer)
    {
        parentContainer->removeElement(this);
    }
    TUIMainWindow::getInstance()->removeElement(this);
}

void TUIElement::setValue(TabletValue type, covise::TokenBuffer &tb)
{
    //cerr << "setValue" << type << endl;
    if (type == TABLET_POS)
    {
        int xp, yp;
        tb >> xp;
        tb >> yp;
        std::string remap = "COVER.TabletUI.Remap:";
        QString qname(getName());
        qname.replace(".", "").replace(":", "");
        remap += qname.toStdString();
        // TODO: won't work for items with identical names but different parents - a random item will be found
        QString parentName;
        if (getParent())
            parentName = getParent()->getName();

#if !defined _WIN32_WCE && !defined ANDROID_TUI
        std::string parent = covise::coCoviseConfig::getEntry("parent", remap);
        if (parent.empty() || parent == parentName.toStdString())
        {
            std::string xpos = covise::coCoviseConfig::getEntry("x", remap);
            if (!xpos.empty())
                xp = atoi(xpos.c_str());
            std::string ypos = covise::coCoviseConfig::getEntry("y", remap);
            if (!ypos.empty())
                yp = atoi(ypos.c_str());
        }
#endif
        setPos(xp, yp);
    }
    else if (type == TABLET_SIZE)
    {
        int x, y;
        tb >> x;
        tb >> y;
        setSize(x, y);
    }
    else if (type == TABLET_LABEL)
    {
        char *l;
        tb >> l;
        QString text = l;
        setLabel(text);
        TUIElement::setLabel(text);
    }
    else if (type == TABLET_COLOR)
    {
        int c;
        tb >> c;
        TUIElement::setColor((Qt::GlobalColor)c);
    }
    else if (type == TABLET_SET_HIDDEN)
    {
        int hide;
        tb >> hide;
        setHidden(hide ? true : false);
        TUIElement::setHidden(hide ? true : false);
    }
    else if (type == TABLET_SET_ENABLED)
    {
        int en;
        tb >> en;
        setEnabled(en ? true : false);
        TUIElement::setEnabled(en ? true : false);
    }
}

/** Set my widget.
  @param w QT Widget
*/
void TUIElement::setWidget(QWidget *w)
{
    widgets.erase(widget);
    widget = w;
    widgets.insert(w);
}

/** Set parent container.
  @param c parent container
*/
void TUIElement::setParent(TUIContainer *c)
{
    parentContainer = c;
    visible = true;
}

/** Set UI element size. Use different values for all dimensions.
  @param xs,ys size = scaling factor for respective dimension (1 is default)
*/
void TUIElement::setSize(int w, int h)
{
    width = w;
    height = h;
}

void TUIElement::setLabel(QString text)
{
    label = text;
}

void TUIElement::setPos(int x, int y)
{
    xPos = x;
    yPos = y;
    TUIContainer *parent = getParent();
    if (parent)
    {
        parent->addElementToLayout(this);
    }
    else
    {
        TUIMainWindow::getInstance()->addElementToLayout(this);
    }
    if (widget)
        widget->setVisible(!hidden);
}

/** Get parent container.
  @return parent container
*/
TUIContainer *TUIElement::getParent()
{
    return parentContainer;
}

QLayout *TUIElement::getLayout()
{
    return layout;
}

/** Set activation state.
  @param en true = element enabled
*/
void TUIElement::setEnabled(bool en)
{
    enabled = en;
    if (getWidget())
        getWidget()->setEnabled(en);
}

/** Set highlight state.
  @param hl true = element highlighted
*/
void TUIElement::setHighlighted(bool hl)
{
    highlighted = hl;
}

void TUIElement::setColor(Qt::GlobalColor color)
{
    if (widget)
    {
        QPalette palette = widget->palette();
        palette.setColor(widget->backgroundRole(), color);
        palette.setColor(widget->foregroundRole(), color);
        widget->setPalette(palette);
    }

    for (auto &w: widgets)
    {
        QPalette palette = w->palette();
        palette.setColor(w->backgroundRole(), color);
        palette.setColor(w->foregroundRole(), color);
        w->setPalette(palette);
    }
}

/** Set element visibility.
  @param newState true = element visible
*/
void TUIElement::setVisible(bool newState)
{
    if (visible == newState)
        return; // state is already ok
    visible = newState;
}

/** hide element
 */
void TUIElement::setHidden(bool hide)
{
    hidden = hide;
    if (getWidget())
        getWidget()->setVisible(!hidden);
    for (auto &w: widgets)
    {
        w->setVisible(!hidden);
    }
}

/** Get hiding state.
  @return hiding state (true = hidden)
*/
bool TUIElement::isHidden() const
{
    return hidden;
}

/** Get activation state.
  @return activation state (true = enabled)
*/
bool TUIElement::isEnabled()
{
    return enabled;
}

/** Get highlight state.
  @return highlight state (true = highlighted)
*/
bool TUIElement::isHighlighted()
{
    return highlighted;
}

/** Get visibility state.
  @return visibility state (true = visible)
*/
bool TUIElement::isVisible()
{
    return visible;
}

const char *TUIElement::getClassName() const
{
    return "TUIElement";
}

QWidget *TUIElement::getWidget()
{
    return widget;
}

bool TUIElement::isOfClassName(const char *classname) const
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
            return false;
            // TUIElement already is the root class. Else:
            // return co::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
