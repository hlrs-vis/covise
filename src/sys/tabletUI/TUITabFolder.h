/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_UI_TAB_FOLDER_H
#define CO_UI_TAB_FOLDER_H

#include <QObject>

#include "TUIContainer.h"

class QTabWidget;

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
    TUITabFolder(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUITabFolder();
    virtual void addElementToLayout(TUIElement *el);

    /// get the Element's classname
    virtual char *getClassName();
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(char *);

public slots:

    void valueChanged(int);

protected:
    QTabWidget *tabWidget;
};
#endif
