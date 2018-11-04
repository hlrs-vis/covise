/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_UI_TAB_FOLDER_H
#define CO_UI_TAB_FOLDER_H

#include <QObject>

#include "TUIContainer.h"

class QTabWidget;
class QStackedWidget;
class QComboBox;

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUITabFolder : public QObject, public TUIContainer
{
    Q_OBJECT

public:
    TUITabFolder(int id, int type, QWidget *w, int parent, QString name, QTabWidget *reuseTabWidget=nullptr);
    virtual ~TUITabFolder();
    virtual void addElementToLayout(TUIElement *el) override;

    /// get the Element's classname
    virtual const char *getClassName() const override;
    int indexOf(QWidget *widget) const;
    void addTab(QWidget *widget, QString label);
    void removeTab(int index);

public slots:

    void valueChanged(int);
    void setCurrentIndex(int);

protected:
    bool deleteTabWidget = true;
    QTabWidget *tabWidget = nullptr;
    QComboBox *switchWidget = nullptr;
    QStackedWidget *stackWidget = nullptr;
};
#endif
