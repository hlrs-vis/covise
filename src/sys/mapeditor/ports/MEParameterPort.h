/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_PARAMETERPORT_H
#define ME_PARAMETERPORT_H

#include "ports/MEPort.h"

namespace covise
{
class coDistributedObject;
}

class QStringList;
class QColor;
class QString;
class QWidget;
class QFrame;
class QPushButton;

class MENodeItem;
class MEControlParameterLine;
class MEModuleParameterLine;
class MEControlParameter;
class MEFileBrowser;
class METimer;

QString toString(double value);
QString toString(const QVariant &value);

//================================================
class MEParameterPort : public MEPort
//================================================
{

    Q_OBJECT

public:
    MEParameterPort(MENode *node, QGraphicsScene *scene, const QString &portname, const QString &paramtype, const QString &description);
    MEParameterPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    enum parameterTypes
    {
        T_STRING,
        T_BOOLEAN,
        T_INT,
        T_FLOAT,
        T_INTVECTOR,
        T_FLOATVECTOR,
        T_CHOICE,
        T_BROWSER,
        T_TABLE,
        T_LIST,
        T_COLORMAP, // only used for
        T_COLOR, // only used for covise
        T_TIMER, // only used for covise
        T_INTSLIDER, // only used for covise
        T_FLOATSLIDER // only used for covise
    };

#ifdef YAC
    enum appearanceTypes
    {
        NOTMAPPED = -1,
        A_STRING,
        A_BOOLEAN,
        A_VECTOR,
        A_LIST,
        A_CHOICE,
        A_BROWSER,
        A_TABLE,
        A_STEPPER,
        A_SLIDER,
        A_SPINBOX,
        A_DIAL
    };
#else
    enum appearanceTypes
    {
        NOTMAPPED = -1,
        NORMAL = 1,
        STEPPER,
        SLIDER
    };
#endif

    enum layoutType
    {
        MODULE = 0,
        CONTROL
    };

    ~MEParameterPort();

    int getAppearance()
    {
        return appearanceType;
    }
    int getParamType();

    bool isEnabled()
    {
        return sensitive;
    }
    bool isMapped()
    {
        return mapped;
    }
    bool hasControlLine();

    void showControlLine();
    void setAppearance(int);
    void setMapped(bool);
    void setShown(bool);
    void setSensitive(bool);
    void sendPanelMessage(const QString &);
    void addItems(bool shown, bool _mapped, int _appearanceType);
    void createParameterLine(QFrame *textFrame, QWidget *contentFrame);
    void sendParamMessage(const QString &value);
    virtual void makeLayout(layoutType, QWidget *) = 0;
    virtual void restoreParam() = 0;
    virtual void storeParam() = 0;
    virtual void defineParam(QString value, int appearanceType);
    virtual void modifyParam(QStringList list, int noOfValues, int istart) = 0;
    virtual void modifyParameter(QString value) = 0;
    virtual void sendExecuteMessage();
    virtual void moduleParameterRequest() = 0;
    QString getParamTypeString();
    QWidget *getSecondLine()
    {
        return secondLine;
    };

    MEControlParameter *getControlInfo();

#ifdef YAC
    virtual void setValues(covise::coRecvBuffer &) = 0;
#endif

public slots:

    void appearanceCB(const QString &);
    void stopCB();
    void left1CB();
    void right1CB();
    void playCB();
    void reverseCB();
    void setFocusCB(bool);

protected:
    bool mapped, sensitive;
    int partype, appearanceType, porttype;

    QString parameterType;
    QWidget *secondLine;
    QPushButton *left1, *left2, *right1, *right2, *stopp;
    QPushButton *folderAction[2];

    METimer *timer;
    MEControlParameterLine *controlLine;
    MEModuleParameterLine *moduleLine;

    void setHelpText();
    virtual void removeFromControlPanel();
    virtual void addToControlPanel();
    virtual void plusNewValue();
    virtual void minusNewValue();

    QColor definePortColor();
};
#endif
