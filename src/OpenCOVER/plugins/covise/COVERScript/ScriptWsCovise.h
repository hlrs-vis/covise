/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SCRIPTWSCOVISE_H
#define SCRIPTWSCOVISE_H

#include <QObject>

#include <wslib/WSCOVISEClient.h>

class ScriptEngineProvider;
using namespace covise;

class ScriptWsCovise : public QObject
{

    Q_OBJECT

    Q_PROPERTY(WSCOVISEClient *covise READ getClient)

public:
    ScriptWsCovise(ScriptEngineProvider *plugin);
    virtual ~ScriptWsCovise();

public slots:
    WSCOVISEClient *getClient() const
    {
        return this->client;
    }

private:
    ScriptEngineProvider *provider;

    WSCOVISEClient *client;
};

#endif // SCRIPTWSCOVISE_H
