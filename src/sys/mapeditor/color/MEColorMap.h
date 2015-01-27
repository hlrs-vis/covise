/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_COLORMAP_H
#define ME_COLORMAP_H

#include <QList>
#include <QFrame>
#include <QWidget>
#include <QPolygon>
#include <QMap>
#include <QPen>

class QPaintEvent;
class QLabel;
class QMouseEvent;
class QKeyEvent;
class QMenu;
class QBoxLayout;
class QStringList;
class QColor;
class QComboBox;
class QString;
class QLineEdit;
class QPushButton;
class QGroupBox;

class MEParameterPort;
class MEColorRGBTable; //
class MEColorChooser;
class MEColorLines;
class MEColorPoint; //
class MEColorMarkerContainer; //
class MEColorMap; //
class MEColorSelector;
class MEHistogram; //

namespace covise
{
class coConfig;
}

extern MEColorMap *colorMap;

//================================================
class MEColorMap : public QFrame
//================================================
{
    friend class MEColorPoint;
    Q_OBJECT

public:
    MEColorMap(MEParameterPort *p = 0, QWidget *parent = 0);
    ~MEColorMap();

    enum type
    {
        MODULE,
        CONTROL
    };

    bool initFlag;

    QList<MEColorPoint *> points;

    QStringList getColorMapNames();
    QStringList values;
    QString getCurrentName()
    {
        return currName;
    };
    QComboBox *getComboBox()
    {
        return namebox;
    };
    MEColorRGBTable *getModulePreview()
    {
        return preview[MODULE];
    };
    MEColorRGBTable *getControlPreview()
    {
        return preview[CONTROL];
    };
    MEColorMarkerContainer *getContainer()
    {
        return container;
    };
    MEColorPoint *getCurrentPoint()
    {
        return currPoint;
    };

    int getNumSteps()
    {
        return points.count();
    };
    float getMin()
    {
        return fmin;
    };
    float getMax()
    {
        return fmax;
    };
    void updateMin(float);
    void updateMax(float);
    void setPredefinedMap(const QString &);
    void getStep(int step, float *r, float *g, float *b, float *a, float *x);
    void updateColorMap(const QString &, const QString &);
    void updateColorMap(int numColors, const float *rgbax);
    void storeCurrentMap();
    void updateHistogram(int np, const float &xmin, const float &xmax, int *values);

private:
    int oldSize;
    float *oldValues;
    QMap<QString, int> mapSize;
    QMap<QString, float *> mapValues;
    float fmin, fmax, currValue;

    QBoxLayout *mainLayout;
    QGroupBox *histo;
    QStringList mapNames;
    QString currName, currType, title, icm;
    QColor currColor;
    QComboBox *namebox;
    QLineEdit *statusline, *tfval;
    QLabel *minval, *maxval;
    QPushButton *saveAs;

    MEParameterPort *port;
    MEColorChooser *chooser;
    MEColorRGBTable *table, *preview[2];
    covise::coConfig *config;
    MEColorMarkerContainer *container;
    MEColorPoint *currPoint;
    MEColorSelector *colorSelector;
    MEHistogram *histogram;

    void makeToolbar();
    void makeEditor();
    void makeButtonPanel();
    void setCurrentMarker();
    void updateAll();
    void deleteMap(const QString &);
    MEColorPoint *makePoint(float pos, QColor);
    void readConfigFile();
    QString getMapValues();

public slots:

    void applyCB();
    void saveXMLCB();
    void removeMarker();
    void resetCB();
    void pointNew(float);
    void showNewColor(const QColor &);
    void pointPicked(MEColorPoint *cp);
    void pointRemoved(MEColorPoint *cp);
    void pointMoved(MEColorPoint *cp);

private slots:

    void loadMap(const QString &name);
    void newtfval();
    void hideCB();

protected:
    void keyPressEvent(QKeyEvent *e);
};

//================================================
class MEColorRGBTable : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEColorRGBTable(MEColorMap *c, QWidget *parent = 0);
    ~MEColorRGBTable();

    void addMarker(MEColorPoint *);

private:
    MEColorMap *colorMap;
    bool preview;
    QPixmap pm_checker;

protected:
    void paintEvent(QPaintEvent *e);

signals:

    void newPoint(float);
};

//================================================
class MEColorMarkerContainer : public QFrame
//================================================
{
    friend class MEColorPoint;
    Q_OBJECT

public:
    MEColorMarkerContainer(MEColorMap *c, QWidget *parent = 0);
    ~MEColorMarkerContainer();

    void addMarker(MEColorPoint *);
    void removeMarker(MEColorPoint *);

private:
    QLineEdit *tfval, *tfmin, *tfmax;
    MEColorMap *colorMap;
    MEColorPoint *clickPoint;
    int offset, ww;

signals:

    void newPoint(float);
    void pickPoint(MEColorPoint *cp);
    void movePoint(MEColorPoint *cp);
    void removePoint();
    void movePoint();
    void sendColor();

protected:
    void mousePressEvent(QMouseEvent *e);
    void mouseMoveEvent(QMouseEvent *e);
    void mouseReleaseEvent(QMouseEvent *e);
    void paintEvent(QPaintEvent *e);
};

//================================================
class MEColorPoint : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEColorPoint(QWidget *p = 0);
    ~MEColorPoint();

    void addItems(float pos, QColor c);
    void setActivated(bool state);
    void setColor(QColor c);
    void setX(float x)
    {
        pos = x;
    }
    QColor getColor()
    {
        return color;
    }
    float getX()
    {
        return pos;
    }
    bool isSelected()
    {
        return state;
    };

private:
    bool state;
    float pos;
    QMenu *menu;
    QColor color;
    QPolygon poly;

signals:

    void removePoint(MEColorPoint *);

private slots:

    void removeCB();

protected:
    void contextMenuEvent(QContextMenuEvent *e);
    void paintEvent(QPaintEvent *e);
};

//================================================
class MEHistogram : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEHistogram(MEColorMap *c, QWidget *parent = 0);
    ~MEHistogram();

    void update(int np, const float &xmin, const float &xmax, int *values);

private:
    int *m_data;
    int m_num;
    float m_fmin, m_fmax;
    MEColorMap *colorMap;
    QPen pen;

protected:
    void paintEvent(QPaintEvent *e);
};

#endif
