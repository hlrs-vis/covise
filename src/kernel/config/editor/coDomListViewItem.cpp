/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef __hpux
#include <rw/stddefs.h>
#endif
#include "coDomListViewItem.h"

#include <config/coConfigLog.h>

const QString DomListViewItem::GSC_NAME("CONFIGEDITOR.COLOR:global_scope");
const QString DomListViewItem::GVC_NAME("CONFIGEDITOR.COLOR:global_variable");
const QString DomListViewItem::HSC_NAME("CONFIGEDITOR.COLOR:host_scope");
const QString DomListViewItem::HVC_NAME("CONFIGEDITOR.COLOR:host_variable");
const QString DomListViewItem::USC_NAME("CONFIGEDITOR.COLOR:user_scope");
const QString DomListViewItem::UVC_NAME("CONFIGEDITOR.COLOR:user_variable");
const QString DomListViewItem::UHSC_NAME("CONFIGEDITOR.COLOR:userhost_scope");
const QString DomListViewItem::UHVC_NAME("CONFIGEDITOR.COLOR:userhost_variable");
const QString DomListViewItem::COL_NAME("color");

DomListViewItem *DomListViewItem::prototype = 0;

DomListViewItem::DomListViewItem(Q3ListView *parent)
    : Q3ListViewItem(parent)
{
}

DomListViewItem::DomListViewItem(Q3ListViewItem *parent)
    : Q3ListViewItem(parent)
{
}

DomListViewItem *DomListViewItem::getInstance(QDomElement node,
                                              Q3ListView *parent)
{

    if (!prototype)
    {
        prototype = new DomListViewItem(parent);
        prototype->updateColors();
        parent->takeItem(prototype);
    }

    DomListViewItem *item = new DomListViewItem(parent);
    item->node = node;

    return item->configureItem();
}

DomListViewItem *DomListViewItem::getInstance(QDomElement node,
                                              Q3ListViewItem *parent)
{

    if (!prototype)
    {
        prototype = new DomListViewItem(parent);
        prototype->updateColors();
        parent->takeItem(prototype);
    }

    DomListViewItem *item = new DomListViewItem(parent);
    item->node = node;

    return item->configureItem();
}

DomListViewItem *DomListViewItem::configureItem()
{

    setColorsFrom(prototype);

    if (node.hasChildNodes())
    {
        setRenameEnabled(1, false);
    }
    else
    {
        setRenameEnabled(1, true);
    }

    if (node.hasAttribute("listitem"))
    {
        setRenameEnabled(1, false);
        setRenameEnabled(0, true);
    }
    else
    {
        setRenameEnabled(0, false);
    }

    setRenameEnabled(2, false);

    return this;
}

DomListViewItem::~DomListViewItem()
{
}

void DomListViewItem::setColorsFrom(const DomListViewItem *colorSource)
{

    globalScopeColor = colorSource->globalScopeColor;
    globalVariableColor = colorSource->globalVariableColor;
    hostScopeColor = colorSource->hostScopeColor;
    hostVariableColor = colorSource->hostVariableColor;
    userScopeColor = colorSource->userScopeColor;
    userVariableColor = colorSource->userVariableColor;
    userHostScopeColor = colorSource->userHostScopeColor;
    userHostVariableColor = colorSource->userHostVariableColor;
}

void DomListViewItem::updateColors()
{

    coConfig *config = coConfig::getInstance();

    globalScopeColor.setNamedColor(config->getValue(COL_NAME, GSC_NAME));
    globalVariableColor.setNamedColor(config->getValue(COL_NAME, GVC_NAME));
    hostScopeColor.setNamedColor(config->getValue(COL_NAME, HSC_NAME));
    hostVariableColor.setNamedColor(config->getValue(COL_NAME, HVC_NAME));
    userScopeColor.setNamedColor(config->getValue(COL_NAME, USC_NAME));
    userVariableColor.setNamedColor(config->getValue(COL_NAME, UVC_NAME));
    userHostScopeColor.setNamedColor(config->getValue(COL_NAME, UHSC_NAME));
    userHostVariableColor.setNamedColor(config->getValue(COL_NAME, UHVC_NAME));
}

DomListViewItem *DomListViewItem::getPrototype()
{
    return prototype;
}

QString DomListViewItem::text(int column) const
{
    switch (column)
    {
    case 0:
        return node.nodeName();
    case 1:
        return node.attribute("value");
    case 2:
        return node.attribute("config");
    case 3:
        return node.attribute("configname");
    default:
        return QString::null;
    }
}

void DomListViewItem::okRename(int column)
{

#if 0
   COCONFIGLOG("DomListViewItem::okRename fixme: disabled");
   return;
#else
    switch (column)
    {
    case 0:
        Q3ListViewItem::okRename(column);
        break;
    case 1:
        Q3ListViewItem::okRename(column);
        coConfig::getInstance()->setValue(node.nodeName(),
                                          node.attribute("value"),
                                          node.attribute("scope"));
        break;
    case 2:
        Q3ListViewItem::okRename(column);
        break;
    }
#endif
}

void DomListViewItem::setText(int column, const QString &text)
{
    if (column == 1)
    {
        node.setAttribute("value", text);
    }
}

void DomListViewItem::paintCell(QPainter *p, const QColorGroup &colorGroup,
                                int column, int width, int align)
{

    QColorGroup cg = colorGroup;

    if (node.attribute("type") == "element")
    {

        if (node.attribute("config") == "global")
        {
            cg.setColor(QColorGroup::Text, globalScopeColor);
        }
        else if (node.attribute("config") == "host")
        {
            cg.setColor(QColorGroup::Text, hostScopeColor);
        }
        else if (node.attribute("config") == "user")
        {
            cg.setColor(QColorGroup::Text, userScopeColor);
        }
        else if (node.attribute("config") == "userhost")
        {
            cg.setColor(QColorGroup::Text, userHostScopeColor);
        }
    }
    else if (node.attribute("type") == "attribute")
    {

        if (node.attribute("config") == "global")
        {
            cg.setColor(QColorGroup::Text, globalVariableColor);
        }
        else if (node.attribute("config") == "host")
        {
            cg.setColor(QColorGroup::Text, hostVariableColor);
        }
        else if (node.attribute("config") == "user")
        {
            cg.setColor(QColorGroup::Text, userVariableColor);
        }
        else if (node.attribute("config") == "userhost")
        {
            cg.setColor(QColorGroup::Text, userHostVariableColor);
        }
    }

    Q3ListViewItem::paintCell(p, cg, column, width, align);
}

QString DomListViewItem::key(int, bool) const
{
    QDomNodeList list = node.parentNode().childNodes();
    for (unsigned int ctr = 0; ctr < list.count(); ctr++)
    {
        if (node == list.item(ctr))
            return QString().setNum(ctr);
    }
    return QString::null;
}

QString DomListViewItem::getScope() const
{
    return node.attribute("scope");
}

QString DomListViewItem::getConfigScope() const
{
    return node.attribute("config");
}

QString DomListViewItem::getConfigName() const
{
    return node.attribute("configname");
}
