/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_UI_TAB_H
#define CO_TUI_UI_TAB_H

#include "TUITab.h"
#include "TUIApplication.h"
#include "TUITextCheck.h"
#include <util/coTabletUIMessages.h>

#include <QObject>

#include <QWidget>

#include "TUIUI/TUIUIWidget.h"

class TUIUITab : public QObject, public TUITab
{
    Q_OBJECT

public:
    TUIUITab(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIUITab();

    virtual const char *getClassName() const override;
    virtual void setValue(TabletValue type, covise::TokenBuffer &tb) override;

public slots:
    void sendCommand(const QString &target, const QString &command);

private:
    QString uiDescription;

    TUIUIWidget *uiWidget;
};
#endif
