/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ScriptWsCovise.h"

#include "ScriptEngineProvider.h"

#ifdef HAVE_CONFIG_H
#undef HAVE_CONFIG_H
#endif

#include <wslib/WSModule.h>
#include <wslib/WSMap.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

#include <appl/CoviseBase.h>
#include <covise/covise_appproc.h>

#define engine this->provider->engine()

using namespace covise;
using namespace opencover;

Q_DECLARE_METATYPE(WSCOVISEClient *)
Q_DECLARE_METATYPE(WSModule *)
Q_DECLARE_METATYPE(WSParameter *)
Q_DECLARE_METATYPE(WSMap *)
Q_DECLARE_METATYPE(QList<WSModule *>)

static QScriptValue wsCOVISEClientToScriptValue(QScriptEngine *eng, WSCOVISEClient *const &in)
{
    return eng->newQObject(const_cast<WSCOVISEClient *>(in));
}

static void wsCOVISEClientFromScriptValue(const QScriptValue &object, WSCOVISEClient *&out)
{
    out = qobject_cast<WSCOVISEClient *>(object.toQObject());
}

static QScriptValue wsModuleListToScriptValue(QScriptEngine *eng, QList<WSModule *> const &in)
{
    QScriptValue list = eng->newArray();
    quint32 ctr = 0;
    foreach (WSModule *module, in)
    {
        list.setProperty(ctr++, qScriptValueFromValue(eng, module));
    }
    return list;
}

static void wsModuleListFromScriptValue(const QScriptValue &object, QList<WSModule *> &out)
{
    out.clear();
    quint32 len = object.property("length").toUInt32();
    for (quint32 ctr = 0; ctr < len; ++ctr)
    {
        QScriptValue item = object.property(ctr);
        out.push_back(qscriptvalue_cast<WSModule *>(item));
    }
}

static QScriptValue wsModuleToScriptValue(QScriptEngine *eng, WSModule *const &in)
{
    return eng->newQObject(const_cast<WSModule *>(in));
}

static void wsModuleFromScriptValue(const QScriptValue &object, WSModule *&out)
{
    out = qobject_cast<WSModule *>(object.toQObject());
}

static QScriptValue wsMapToScriptValue(QScriptEngine *eng, WSMap *const &in)
{
    return eng->newQObject(const_cast<WSMap *>(in));
}

static void wsMapFromScriptValue(const QScriptValue &object, WSMap *&out)
{
    out = qobject_cast<WSMap *>(object.toQObject());
}

static QScriptValue wsParameterToScriptValue(QScriptEngine *eng, WSParameter *const &in)
{
    return eng->newQObject(const_cast<WSParameter *>(in));
}

static void wsParameterFromScriptValue(const QScriptValue &object, WSParameter *&out)
{
    out = qobject_cast<WSParameter *>(object.toQObject());
}

ScriptWsCovise::ScriptWsCovise(ScriptEngineProvider *provider)
    : provider(provider)
    , client(new covise::WSCOVISEClient())
{
    qScriptRegisterMetaType(&engine, wsCOVISEClientToScriptValue, wsCOVISEClientFromScriptValue);
    qScriptRegisterMetaType(&engine, wsModuleListToScriptValue, wsModuleListFromScriptValue);
    qScriptRegisterMetaType(&engine, wsModuleToScriptValue, wsModuleFromScriptValue);
    qScriptRegisterMetaType(&engine, wsMapToScriptValue, wsMapFromScriptValue);
    qScriptRegisterMetaType(&engine, wsParameterToScriptValue, wsParameterFromScriptValue);
    engine.globalObject().setProperty("covise", engine.newQObject(client));
    // FIXME event handling. Events are consumed directly by DynamicUI
    client->setEventsAsSignal(true, true);

    QString endpoint;

    if (!OpenCOVER::instance()->visPlugin())
    {
        endpoint = "http://localhost:31090/"; // Doesn't really make sense.... but maybe a session is running
    }
    else
    {
        if (opencover::coVRMSController::instance()->isMaster())
        {
            QString controllerHost = CoviseBase::appmod->getControllerConnection()->get_hostname();
            endpoint = "http://" + controllerHost + ":31090/";
            int size = endpoint.size() + 1;
            opencover::coVRMSController::instance()->sendSlaves(&size, sizeof(int));
            opencover::coVRMSController::instance()->sendSlaves(endpoint.toLatin1().data(), endpoint.size() + 1);
            client->setReadOnly(false);
        }
        else
        {
            int size;
            opencover::coVRMSController::instance()->readMaster(&size, sizeof(int));
            char *endpointData = new char[size];
            opencover::coVRMSController::instance()->readMaster(endpointData, size);
            endpoint = endpointData;
            delete[] endpointData;
            client->setReadOnly(true);
        }
    }

    std::cerr << "ScriptWsCovise::<init> info: using controller at " << qPrintable(endpoint) << std::endl;

    client->attach(endpoint);
}

ScriptWsCovise::~ScriptWsCovise()
{
    engine.globalObject().setProperty("covise", engine.undefinedValue());
    delete this->client;
}
