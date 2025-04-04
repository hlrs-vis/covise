/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** \mainpage tabletUI
  tabletUI is a simple GUI Glient for COVER based on QT.
*/

#ifndef TABLET_UI_ELEMENT_H
#define TABLET_UI_ELEMENT_H

#include <QString>
#include <QObject>
#include <net/covise_connect.h>
#include <util/coTabletUIMessages.h>
#include <set>
#include "export.h"
class TUIContainer;
class QWidget;
class QLayout;
class TUITab;
class QGridLayout;
namespace covise
{
class TokenBuffer;
}

/**
 * Basic TUI GUI element.
 * This class provides functionality for all TUI elements like position,
 * size, availability, parent, etc.<BR>
 * At least this class should be subclassed for any new GUI element types.<BR>
 * All inheritable functions are defined virtual so that they can be overwritten
 * by subclasses.
 */
class TUIEXPORT TUIElement
{
private:
    TUIContainer *parentContainer; ///< info about parent container, needed by layout

public:
    TUIElement(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIElement();
    virtual void setParent(TUIContainer *);
    virtual TUIContainer *getParent();
    virtual QGridLayout *getLayout();
    virtual QWidget *getWidget();
    virtual void setWidget(QWidget *);

    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);
    virtual void setColor(Qt::GlobalColor color);
    virtual void setHidden(bool hidden);
    virtual bool isEnabled();
    virtual bool isHighlighted();
    virtual bool isHidden() const;
    virtual void deActivate(TUITab *activedTab)
    {
        (void)activedTab;
    };
    virtual int getXpos() const ///< Returns element x position
    {
        return xPos;
    };
    virtual int getYpos()  const ///< Returns element y position
    {
        return yPos;
    };
    virtual void setPos(int, int); ///< Set element location.
    virtual void setSize(int w = 1, int h = 1);
    virtual int getWidth() const
    {
        return width;
    };
    virtual int getHeight() const
    {
        return height;
    };
    virtual void setValue(TabletValue type, covise::TokenBuffer &);
    virtual void setLabel(QString textl);
    virtual QString getLabel() const
    {
        return label;
    };
    virtual QString getName() const
    {
        return name;
    };
    int getID() const
    {
        return ID;
    };
    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    int ID;
    int xPos, yPos;
    int height = 1, width = 1;
    QString label;
    std::set<QWidget *> widgets;
    bool enabled; ///< true if UI element is enabled, false if UI element cannot be used
    bool highlighted; ///< true if highlighted
    bool hidden; ///< true if UI element is to be hidden at any time
    QString name;
    
    QWidget *widget();
    QGridLayout *createLayout(QWidget *parent);

    template<typename T>
    T *createWidget(QWidget *parent)
    {
        deleteWidget();

        widgetHasParent = parent;
        auto t = new T(parent);
        m_widget = t;
        QObject::connect(m_widget, &QObject::destroyed, [this](QObject*){
            m_widget = nullptr;
        });
        return t;
    }

    void setWidget(QWidget *widget, bool hasParent);



private:
    QGridLayout *layout = nullptr;
    bool layoutHasParent = true;
    QWidget *m_widget = nullptr;
    bool widgetHasParent = true;
    void deleteWidget();
    

};
#endif

