/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SCRIPTTABLETUI_H
#define SCRIPTTABLETUI_H
#include <util/coTypes.h>
#include <QObject>
#include <QtScript>

class ScriptEngineProvider;

namespace opencover
{
class coTUITabFolder;
}

using namespace opencover;

class ScriptTabletUI : public QObject
{

    Q_OBJECT

    Q_PROPERTY(coTUITabFolder *mainpanel READ getMainPanel)

public:
    ScriptTabletUI(ScriptEngineProvider *provider);
    virtual ~ScriptTabletUI();

signals:

public slots:
    opencover::coTUITabFolder *getMainPanel();

private:
    ScriptEngineProvider *provider;

    template <class T>
    void registerClass(const QString &className);
    template <class T>
    void registerClass(const QString &className, QScriptEngine::FunctionSignature function);

    static ScriptTabletUI *instance;
    template <class T>
    static QScriptValue simple_ctor(QScriptContext *context, QScriptEngine *engine);
    template <class T>
    static QScriptValue sib_ctor(QScriptContext *context, QScriptEngine *engine);
    template <class T>
    static QScriptValue ssib_ctor(QScriptContext *context, QScriptEngine *engine);
    template <class T>
    static QScriptValue sii_ctor(QScriptContext *context, QScriptEngine *engine);
    template <class T>
    static QScriptValue sif_ctor(QScriptContext *context, QScriptEngine *engine);
};

#endif // SCRIPTTABLETUI_H
