/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SCRIPTENGINEPROVIDER_H
#define SCRIPTENGINEPROVIDER_H

#include <QScriptEngine>

class ScriptEngineProvider
{
public:
    ScriptEngineProvider();
    virtual ~ScriptEngineProvider();

    virtual int loadScript(const QString &file) = 0;
    virtual QScriptEngine &engine() = 0;
};

#endif // SCRIPTENGINEPROVIDER_H
