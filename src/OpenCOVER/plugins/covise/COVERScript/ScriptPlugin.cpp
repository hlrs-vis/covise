/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: Script Plugin (does nothing)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Nov-01  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "ScriptPlugin.h"
#include "ScriptTabletUI.h"
#include "ScriptWsCovise.h"
#include "ScriptVrmlNode.h"

#if QT_VERSION >= 0x040500
#include "ScriptDebugger.h"
#endif

#include <net/tokenbuffer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRTui.h>
#include <PluginUtil/PluginMessageTypes.h>

#include <config/CoviseConfig.h>
#include <QFile>
#include <QTextStream>

#include <QApplication>
#include <QMainWindow>

#include <QPushButton>
#include <net/tokenbuffer.h>

ScriptPlugin *ScriptPlugin::plugin = NULL;
static FileHandler handlers[] = {
    { NULL,
      ScriptPlugin::loadQS,
      ScriptPlugin::loadQS,
      ScriptPlugin::unloadQS,
      "qs" },
    { NULL,
      ScriptPlugin::loadQS,
      ScriptPlugin::loadQS,
      ScriptPlugin::unloadQS,
      "js" }
};

static int argc = 0;
static char *argv[1] = { 0 };

ScriptPlugin::ScriptPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "\nScriptPlugin::ScriptPlugin (with %s QApplication instance)\n",
            (qApp == 0 ? "new" : "old"));

    if (qApp == 0)
    {
        app = new QApplication(argc, argv);
        app->setAttribute(Qt::AA_MacDontSwapCtrlAndMeta);
    }
    else
    {
        app = qApp;
    }

    plugin = this;

    this->engine_ = new QScriptEngine();

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    coVRFileManager::instance()->registerFileHandler(&handlers[1]);
    //fprintf(stderr,"ScriptPlugin::ScriptPlugin\n");
}

// this is called if the plugin is removed at runtime
ScriptPlugin::~ScriptPlugin()
{
    fprintf(stderr, "ScriptPlugin::~ScriptPlugin\n");
    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[1]);
    delete this->commandLine;
    delete this->commandLineLabel;
    delete this->myCOVER;
    delete this->tui;
    delete this->dynamicUI;
    delete this->covise_;
#if QT_VERSION >= 0x040500
    delete this->debugger;
#endif
}

int ScriptPlugin::unloadQS(const char *filename, const char * /*covise_key*/)
{
    return plugin->unloadScript(filename);
}

int ScriptPlugin::unloadScript(const QString & /*filename*/)
{
    return 1;
}

int ScriptPlugin::loadQS(const char *filename, osg::Group * /*loadParent*/, const char * /*covise_key*/)
{
    return plugin->loadScript(filename);
}

int ScriptPlugin::loadScript(const QString &fileName)
{
    QFile scriptFile(fileName);
    if (!scriptFile.open(QIODevice::ReadOnly))
    {
        return 0;
    }

    std::cerr << "ScriptPlugin::loadScript info: loading script file " << qPrintable(fileName) << std::endl;

    QTextStream stream(&scriptFile);
    QString contents = stream.readAll();
    scriptFile.close();
    engine().evaluate(contents, fileName);

    return 1;
}

QScriptEngine &ScriptPlugin::engine()
{
    return *(this->engine_);
}

void ScriptPlugin::evaluate(const QString &command)
{
    if (std::cout.bad())
        std::cout.clear();
    engine().evaluate(command);
}

void ScriptPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == this->commandLine)
    {
        evaluate(QString::fromStdString(this->commandLine->getText()));
    }
}

bool ScriptPlugin::init()
{
    fprintf(stderr, "ScriptPlugin::init\n");
    QString startupScript = QString::fromStdString(coCoviseConfig::getEntry("COVER.Plugin.COVERScript.StartupScript"));
    if (startupScript != "")
    {
        loadScript(startupScript);
    }

    this->commandLine = new coTUIEditField("cover.setVisible(\"Sphere01-FACES\",true);", coVRTui::instance()->getCOVERTab()->getID());
    this->commandLine->setPos(1, 13);
    this->commandLine->setEventListener(this);
    this->commandLineLabel = new coTUILabel("CommandLine", coVRTui::instance()->getCOVERTab()->getID());
    this->commandLineLabel->setPos(0, 13);
    this->myCOVER = new ScriptInterface(this);
    QScriptValue objectValue = engine().newQObject(myCOVER);
    engine().globalObject().setProperty("cover", objectValue);
    engine().setProcessEventsInterval(1);

    this->tui = new ScriptTabletUI(this);
    this->covise_ = new ScriptWsCovise(this);
    this->dynamicUI = new DynamicUI(this, this->covise_);

#if QT_VERSION >= 0x040500
    this->debugger = new ScriptDebugger(this);

    if (coCoviseConfig::isOn("COVER.Plugin.COVERScript.ShowDebugger", false))
        this->debugger->show();
#endif
    ScriptVrmlNode::init(this);
    return true;
}

void ScriptPlugin::preFrame()
{
    if (std::cout.bad())
        std::cout.clear();
    this->engine_->collectGarbage();
    this->dynamicUI->preFrame();
    this->app->processEvents();
}

void ScriptPlugin::message(int type, int len, const void *buf)
{

    switch (type)
    {
    case opencover::PluginMessageTypes::COVERScriptEvaluate:
    {
        TokenBuffer tb((const char *)buf, len);
        int size;
        ushort *data;
        tb >> size;
        data = (ushort *)tb.getBinary(size);

        QString command = QString::fromUtf16(data);
        //std::cerr << "ScriptPlugin::message info: evaluate(" << qPrintable(command) << ")" << std::endl;
        evaluate(command);
    }
    default:
        break;
    }
}

COVERPLUGIN(ScriptPlugin)
