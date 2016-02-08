/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSLINK_H
#define WSLINK_H

#include <QObject>

#include "WSExport.h"
#include "WSCoviseStub.h"

namespace covise
{

class WSPort;

class WSLIBEXPORT WSLink : public QObject
{
    Q_OBJECT
public:
    WSLink(WSPort *from, WSPort *to);
    virtual ~WSLink();

    static QString makeID(WSPort *from, WSPort *to);
    static QString makeID(const QString &fromModule, const QString &fromPort, const QString &toModule, const QString &toPort);

    virtual covise__Link getSerialisable() const;

signals:
    void deleted(const QString &linkID);

public slots:
    WSPort *from() const;
    WSPort *to() const;
    bool isLinkTo(const WSPort *port) const;
    const QString &getLinkID() const;

private slots:
    void portDeleted();

private:
    WSPort *fromPort;
    WSPort *toPort;

    QString id;
};
}
#endif // WSLINK_H
