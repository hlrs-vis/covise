/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_ThyssenButton_H
#define CO_ThyssenButton_H

#include <QObject>

#include "TUIElement.h"

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class ThyssenButton : public QObject, public TUIElement
{
    Q_OBJECT
    int number=0;
    bool oldState=false;
public:
    ThyssenButton(int id, int type, QWidget *w, int parent, QString name);
    virtual ~ThyssenButton();
    virtual void setSize(int w, int h) override;
    virtual void setLabel(QString textl) override;

    /// get the Element's classname
    virtual const char *getClassName() const override;
    virtual void update(uint8_t bs,bool wasPressed);
    int getNumber(){return number;};

public slots:

    void pressed();
    void released();

protected:
};
#endif
