/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ScriptTabletUI.h"

#include "ScriptEngineProvider.h"

#include <QtScript>

#include <cover/coVRTui.h>

#define engine this->provider->engine()

ScriptTabletUI *ScriptTabletUI::instance = 0;

Q_DECLARE_METATYPE(opencover::coTUITabFolder *)

//static QScriptValue getMainPanelForScript(QScriptContext *, QScriptEngine *);
static QScriptValue coTUITabFolderToScriptValue(QScriptEngine *, opencover::coTUITabFolder *const &);
static void coTUITabFolderFromScriptValue(const QScriptValue &, opencover::coTUITabFolder *&);

static QScriptValue stdStringToScriptValue(QScriptEngine *, std::string const &);
static void stdStringFromScriptValue(const QScriptValue &, std::string &);

template <class T>
QScriptValue ScriptTabletUI::simple_ctor(QScriptContext *ctx, QScriptEngine *eng)
{
    if (!ctx->isCalledAsConstructor())
        return ctx->throwError(QScriptContext::SyntaxError, "please use the 'new' operator");

    QString arg1 = qscriptvalue_cast<QString>(ctx->argument(0));
    int arg2 = qscriptvalue_cast<int>(ctx->argument(1));
    T *t = new T(instance, arg1.toStdString(), arg2);

    return eng->newQObject(ctx->thisObject(), t, QScriptEngine::ScriptOwnership);
}

template <class T>
QScriptValue ScriptTabletUI::sib_ctor(QScriptContext *ctx, QScriptEngine *eng)
{
    if (!ctx->isCalledAsConstructor())
        return ctx->throwError(QScriptContext::SyntaxError, "please use the 'new' operator");

    QString arg1 = qscriptvalue_cast<QString>(ctx->argument(0));
    int arg2 = qscriptvalue_cast<int>(ctx->argument(1));
    bool arg3 = false;
    if (ctx->argumentCount() > 2)
        arg3 = qscriptvalue_cast<bool>(ctx->argument(2));

    T *t = new T(instance, arg1.toStdString(), arg2, arg3);

    return eng->newQObject(ctx->thisObject(), t, QScriptEngine::ScriptOwnership);
}

template <class T>
QScriptValue ScriptTabletUI::sii_ctor(QScriptContext *ctx, QScriptEngine *eng)
{
    if (!ctx->isCalledAsConstructor())
        return ctx->throwError(QScriptContext::SyntaxError, "please use the 'new' operator");

    QString arg1 = qscriptvalue_cast<QString>(ctx->argument(0));
    int arg2 = qscriptvalue_cast<int>(ctx->argument(1));
    int arg3 = 0;
    if (ctx->argumentCount() > 2)
        arg3 = qscriptvalue_cast<int>(ctx->argument(2));

    T *t = new T(instance, arg1.toStdString(), arg2, arg3);

    return eng->newQObject(ctx->thisObject(), t, QScriptEngine::ScriptOwnership);
}

template <class T>
QScriptValue ScriptTabletUI::sif_ctor(QScriptContext *ctx, QScriptEngine *eng)
{
    if (!ctx->isCalledAsConstructor())
        return ctx->throwError(QScriptContext::SyntaxError, "please use the 'new' operator");

    QString arg1 = qscriptvalue_cast<QString>(ctx->argument(0));
    int arg2 = qscriptvalue_cast<int>(ctx->argument(1));
    float arg3 = 0;
    if (ctx->argumentCount() > 2)
        arg3 = qscriptvalue_cast<float>(ctx->argument(2));

    T *t = new T(instance, arg1.toStdString(), arg2, arg3);

    return eng->newQObject(ctx->thisObject(), t, QScriptEngine::ScriptOwnership);
}

template <class T>
QScriptValue ScriptTabletUI::ssib_ctor(QScriptContext *ctx, QScriptEngine *eng)
{
    if (!ctx->isCalledAsConstructor())
        return ctx->throwError(QScriptContext::SyntaxError, "please use the 'new' operator");

    QString arg1 = qscriptvalue_cast<QString>(ctx->argument(0));
    QString arg2 = qscriptvalue_cast<QString>(ctx->argument(1));
    int arg3 = qscriptvalue_cast<int>(ctx->argument(2));
    bool arg4 = false;
    if (ctx->argumentCount() > 3)
        arg4 = qscriptvalue_cast<bool>(ctx->argument(3));

    T *t = new T(instance, arg1.toStdString(), arg2.toStdString(), arg3, arg4);

    return eng->newQObject(ctx->thisObject(), t, QScriptEngine::ScriptOwnership);
}

template <class T>
void ScriptTabletUI::registerClass(const QString &className, QScriptEngine::FunctionSignature function)
{
    QScriptValue tuiClass = engine.scriptValueFromQMetaObject<T>();
    engine.globalObject().setProperty(className, tuiClass);

    QScriptValue ctor = engine.newFunction(function);
    QScriptValue metaObject = engine.newQMetaObject(&T::staticMetaObject, ctor);
    engine.globalObject().setProperty(className, metaObject);
}

template <class T>
void ScriptTabletUI::registerClass(const QString &className)
{
    registerClass<T>(className, simple_ctor<T>);
}

ScriptTabletUI::ScriptTabletUI(ScriptEngineProvider *provider)
{

    ScriptTabletUI::instance = this;

    this->provider = provider;

    qScriptRegisterMetaType(&engine, coTUITabFolderToScriptValue, coTUITabFolderFromScriptValue);
    qScriptRegisterMetaType(&engine, stdStringToScriptValue, stdStringFromScriptValue);

    engine.globalObject().setProperty("tui", engine.newQObject(this));

    registerClass<opencover::coTUITab>("coTUITab");
    registerClass<opencover::coTUILabel>("coTUILabel");
    registerClass<opencover::coTUIBitmapButton>("coTUIBitmapButton");
    registerClass<opencover::coTUIButton>("coTUIButton");
    registerClass<opencover::coTUIColorTriangle>("coTUIColorTriangle");
    registerClass<opencover::coTUIColorButton>("coTUIColorButton");
    registerClass<opencover::coTUIColorTab>("coTUIColorTab");
    registerClass<opencover::coTUISplitter>("coTUISplitter");
    registerClass<opencover::coTUIFrame>("coTUIFrame");
    registerClass<opencover::coTUITabFolder>("coTUITabFolder");
    registerClass<opencover::coTUIMessageBox>("coTUIMessageBox");
    registerClass<opencover::coTUIProgressBar>("coTUIProgressBar");
    registerClass<opencover::coTUIFloatSlider>("coTUIFloatSlider");
    registerClass<opencover::coTUISlider>("coTUISlider");
    registerClass<opencover::coTUISpinEditfield>("coTUISpinEditfield");
    registerClass<opencover::coTUITextSpinEditField>("coTUITextSpinEditField");
    registerClass<opencover::coTUIEditField>("coTUIEditField");
    registerClass<opencover::coTUIEditTextField>("coTUIEditTextField");
    registerClass<opencover::coTUIComboBox>("coTUIComboBox");
    registerClass<opencover::coTUIListBox>("coTUIListBox");
    registerClass<opencover::coTUIPopUp>("coTUIPopUp");

    //  NOT YET
    //registerClass<opencover::coTUIFileBrowserButton>("coTUIFileBrowserButton");
    //registerClass<opencover::coTUIFunctionEditorTab>("coTUIFunctionEditorTab");
    //registerClass<opencover::coTUITextureTab>("coTUITextureTab");
    //registerClass<opencover::coTUISGBrowserTab>("coTUISGBrowserTab");
    //registerClass<opencover::coTUIAnnotationTab>("coTUIAnnotationTab");
    //registerClass<opencover::coTUINav>("coTUINav");
    //registerClass<opencover::coTUIMap>("coTUIMap");

    //Nonsimple Ctor
    registerClass<opencover::coTUIToggleButton>("coTUIToggleButton", sib_ctor<opencover::coTUIToggleButton>);
    registerClass<opencover::coTUIToggleBitmapButton>("coTUIToggleBitmapButton", ssib_ctor<opencover::coTUIToggleBitmapButton>);
    registerClass<opencover::coTUIEditIntField>("coTUIEditIntField", sii_ctor<opencover::coTUIEditIntField>);
    registerClass<opencover::coTUIEditFloatField>("coTUIEditFloatField", sif_ctor<opencover::coTUIEditFloatField>);
}

ScriptTabletUI::~ScriptTabletUI()
{
    engine.globalObject().setProperty("tui", engine.undefinedValue());
}

opencover::coTUITabFolder *ScriptTabletUI::getMainPanel()
{
    return opencover::coVRTui::instance()->mainFolder;
}

QScriptValue coTUITabFolderToScriptValue(QScriptEngine *eng, opencover::coTUITabFolder *const &in)
{
    return eng->newQObject(const_cast<opencover::coTUITabFolder *>(in));
}

void coTUITabFolderFromScriptValue(const QScriptValue &object, opencover::coTUITabFolder *&out)
{
    out = qobject_cast<opencover::coTUITabFolder *>(object.toQObject());
}

QScriptValue stdStringToScriptValue(QScriptEngine *, std::string const &in)
{
    return QScriptValue(QString::fromStdString(in));
}

void stdStringFromScriptValue(const QScriptValue &object, std::string &out)
{
    out = object.toString().toStdString();
}
