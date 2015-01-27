/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef DOMLISTVIEWITEM_H
#define DOMLISTVIEWITEM_H

#include <qcolor.h>
#include <qdom.h>
#include <q3listview.h>
#include <qstring.h>

#include <config/coConfig.h>

class CONFIGEDITOREXPORT DomListViewItem : public Q3ListViewItem
{

public:
    static DomListViewItem *getInstance(QDomElement node, Q3ListView *parent);
    static DomListViewItem *getInstance(QDomElement node, Q3ListViewItem *parent);
    ~DomListViewItem();

    DomListViewItem *getPrototype();
    void updateColors();

    virtual void setText(int column, const QString &text);
    virtual QString text(int column) const;

    virtual void paintCell(QPainter *p, const QColorGroup &cg,
                           int column, int width, int align);
    virtual QString key(int, bool) const;

    virtual QString getScope() const;
    virtual QString getConfigScope() const;
    virtual QString getConfigName() const;

protected:
    virtual void okRename(int col);

private:
    DomListViewItem(Q3ListView *parent);
    DomListViewItem(Q3ListViewItem *parent);

    void setColorsFrom(const DomListViewItem *colorSource);
    DomListViewItem *configureItem();

    static DomListViewItem *prototype;

    QDomElement node;

    QColor globalScopeColor;
    QColor globalVariableColor;
    QColor hostScopeColor;
    QColor hostVariableColor;
    QColor userScopeColor;
    QColor userVariableColor;
    QColor userHostScopeColor;
    QColor userHostVariableColor;

    static const QString GSC_NAME;
    static const QString GVC_NAME;
    static const QString HSC_NAME;
    static const QString HVC_NAME;
    static const QString USC_NAME;
    static const QString UVC_NAME;
    static const QString UHSC_NAME;
    static const QString UHVC_NAME;

    static const QString COL_NAME;
};

#endif
