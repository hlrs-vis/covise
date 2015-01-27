/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_FUNCEDIT_TAB_H
#define CO_TUI_FUNCEDIT_TAB_H

//#include <util/coTypes.h>
#include "TUITab.h"
#include "TUILabel.h"
#include "TUIColorWidget.h"
#include "qtcolortriangle.h"
#include "TUITFEditor.h"
#include "TUITFEWidgets.h"

#include <QObject>
#include <QGroupBox>

#include <QList>
#include <QFrame>
#include <QWidget>
#include <QPolygon>
#include <QMap>
#include <QPen>
#include <QMenu>
#include <QLabel>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QComboBox>
#include <QPushButton>
#include <QCheckBox>
#include <QLineEdit>
#include <QPainter>
#include <QHBoxLayout>
#include <QDebug>
#include <QGroupBox>
#include <QInputDialog>
#include <QDir>

class TUIFunctionEditorTab;

//================================================
class TUIHistogram : public QWidget
{
    Q_OBJECT

public:
    TUIHistogram(TUIFunctionEditorTab *c, QWidget *parent = 0);
    ~TUIHistogram();

    void update(int np, const float &xmin, const float &xmax, int *values);
    void computeLogData();

private:
    float *_logdata;
    int *m_data;
    int m_num;
    float m_fmin, m_fmax;
    TUIFunctionEditorTab *colorMap;
    QPen histoPen;
    QBrush histoBrush;
    QBrush histoLogBrush;

protected:
    void paintEvent(QPaintEvent *e);
};

//
// A tab in the TabletUI interface to visualize and modify the transfer function
// for volumetric rendering. Initialized by the Volume plugin
// Step1: use the transfer function editor of the mapeditor (ME)
//
//QFrame?
class TUIFunctionEditorTab : public QObject, public TUITab
{
    friend class TUIColorPoint;
    Q_OBJECT

public: // data structures
    enum TFEditorType
    {
        TF_1D_EDITOR = 1,
        TF_2D_EDITOR = 2,
    };

public:
    //
    // Constructor / desctructor
    //
    TUIFunctionEditorTab(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIFunctionEditorTab();

    //
    // TUI interface
    //
    virtual char *getClassName();
    virtual void setValue(int type, covise::TokenBuffer &tb);

    // To update the cover counterpart
public slots:
    void valueChanged();

public: // data structures
    //
    // Transfer function data
    //
    bool initFlag;
    bool userDefined;

private: // data structures
    float fmin, fmax, currValue;

    //Layouts
    QBoxLayout *mainLayout;
    QGroupBox *gb;
    QGroupBox *histo;

    QStringList mapNames;
    QString currName, currType, title;
    QString udc, icm;
    QColor currColor;
    QComboBox *namebox, *nameboxM, *nameboxC;
    QLineEdit *statusline, *tfval;
    QLabel *minval, *maxval;
    QPushButton *saveAs;

    // Command bar
    QPushButton *drawAlphaFree;
    QPushButton *eraseAlphaFree;
    QPushButton *deleteMarker;
    QPushButton *applyFunction;
    QLineEdit *editAlpha;
    QCheckBox *chkOwnColor;

    QPushButton *btn1D;
    QPushButton *btn2D;

    TUIElement *parent;

    // Editor objects and widgets
    TFEditorType editorType;
    TUITFEditor *functionEditor;
    TUIHistogram *histogram;
    TUIColorWidget *chooser;

    void makeToolbar();
    void makeEditor();
    void make1DEditor();
    void make2DEditor();

    void makeButtonPanel();
    void setCurrentMarker();
    void updateAll();
    void showNewPoint();
    void createUserMap();
    //void                   deleteColorMap     (const QString &);
    void storeCurrentColorMap();
    void readConfigFile();
    QString getMapValues();

public:
    void setFocus()
    {
        if (widget != NULL)
            widget->setFocus();
    }

    // TUI message parsers
    void widgetListMessage(covise::TokenBuffer &tb);
    void histogramMessage(covise::TokenBuffer &tb);

public:
    QStringList getColorMapNames();
    QStringList values;
    QString getCurrentName()
    {
        return currName;
    }

    //int                    getNumSteps           () {return colorPoints.count() + alphaPoints.count(); }
    float getMin()
    {
        return fmin;
    }
    float getMax()
    {
        return fmax;
    }
    QColor getCurrentColor()
    {
        return chooser->getColor();
    }
    void removeMarker();
    void updateMin(float);
    void updateMax(float);
    void updateHistogram(int np, const float &xmin, const float &xmax, int *values);

public slots:

    void pointPicked(TUITFEWidget *wp);
    void pointRemoved(TUITFEWidget *cp);
    void pointAdded(TUITFEWidget *cp);
    void valuesChanged(TUITFEWidget *cp);

    void showNewColor(QColor);
    void changeTFEDimension(int newType);

private slots:
    void newWidgetValue();
    void changeWidgetType(int);
    void changedAlphaValue(const QString &);

protected:
    void keyPressEvent(QKeyEvent *e);
};
#endif //CO_TUI_FUNCEDIT_TAB_H
