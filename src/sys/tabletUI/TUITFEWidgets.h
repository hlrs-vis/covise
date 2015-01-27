/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TUI_TFE_WIDGETS_H_INCLUDED
#define TUI_TFE_WIDGETS_H_INCLUDED

#include <QColor>
#include <QPainter>
#include <QImage>

#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <virvo/vvtransfunc.h>
#else
#endif

class TUITFEWidget
{
public:
    // Widgets in the first set must have the same values
    // and types of virvo TF widgets!
    enum TFKind
    {
        TF_COLOR,
        TF_PYRAMID,
        TF_BELL,
        TF_SKIP,
        TF_CUSTOM,
        TF_CUSTOM_2D,
        TF_MAP,
        TF_CUSTOM_2D_EXTRUDE = 11,
        TF_CUSTOM_2D_TENT = 12
    };

    enum HandleType
    {
        HT_TOP,
        HT_BOTTOM,
        HT_MIDDLE,
        HT_NONE
    };

    TUITFEWidget(TFKind kind)
        : modified(true)
    {
        this->kind = kind;
        selectedHandle = HT_NONE;
    }

    virtual ~TUITFEWidget()
    {
    }

    void setModified(bool mod = true)
    {
        modified = mod;
    }

    bool isModified() const
    {
        return modified;
    }

    virtual void setActivated()
    {
        selectedHandle = HT_NONE;
    }

    virtual void setX(float x)
    {
        pos = x;
    }
    virtual float getX()
    {
        return pos;
    }
    virtual QColor getColor()
    {
        return QColor(Qt::blue);
    }
    virtual bool hasOwnColor()
    {
        return false;
    }
    virtual void setColor(QColor)
    {
    }

    TFKind getKind()
    {
        return kind;
    }

protected:
    HandleType selectedHandle;
    float pos;
    TFKind kind;
    bool modified;

public:
    virtual void paint(QPainter &p, int w, bool active) = 0;
    virtual bool contains(int x, int y, int w) = 0;

    virtual HandleType testHit(int /*x*/, int /*y*/)
    {
        return selectedHandle;
    }
    HandleType getSelectedHandle()
    {
        return selectedHandle;
    }
    TFKind getWidgetType()
    {
        return kind;
    }
};

//================================================
class TUIAlphaTriangle : public TUITFEWidget
{
public:
    TUIAlphaTriangle();

    //ramp is defined by: position, ramp base (can be larger than one)
    //ramp top (can be 0, in which case is a triangle)
    void setData(float pos, int alpha, float xb, float xt);
    void setX(float x);
    void setXb(float xb);
    void setXt(float xt);
    void setAlpha(int alpha);
    int getAlpha()
    {
        return alpha;
    }
    float getXb()
    {
        return xb;
    }
    float getXt()
    {
        return xt;
    }

private:
    int alpha;
    float xb, xt;

    int ww;
    QImage alphaMap;

    void updateAlphaMap();

public:
    void paint(QPainter &p, int w, bool active);
    bool contains(int x, int y, int w);
    HandleType testHit(int x, int y);
};

//================================================
class TUIColorPoint : public TUITFEWidget
{
public:
    TUIColorPoint();

    void setData(float pos, QColor c);
    void setColor(QColor c);
    QColor getColor()
    {
        return color;
    }
    bool hasOwnColor()
    {
        return true;
    }

private:
    QColor color;

public:
    void paint(QPainter &p, int w, bool active);
    bool contains(int x, int y, int w);
};

//================================================
class TUIAlphaFree
{
public:
    static const int alphaMapSize = 256;

    TUIAlphaFree();

    void addLevel(float xPos, float yPos);
    void removeLevel(float xPos, float yPos);
    int getAlpha(float xPos);
    void erase();
    bool isNotEmpty()
    {
        return hasData;
    }

    // we expect the buffer pointed by data to be of 'alphaMapSize' lenght
    void toData(float *data);
    bool fromData(float *data, int count);

    void beginDraw();
    void endDraw();

private:
    bool drawing;
    int lastX;
    int lastY;

    void addAlphaInternal(int currX, int currY);
    void removeAlphaInternal(int currX, int currY);

    bool hasData;
    int ww;
    QImage alphaMap;

public:
    void paint(QPainter &p, int w, bool active);
};

//
// widgets linked to virvo widgets
//

class TUIVirvoWidget : public TUITFEWidget
{
private:
    vvTransFunc *tf;

public:
    friend class Canvas;

    vvTFWidget *vvWidget;

public:
    TUIVirvoWidget(TFKind myKind, QColor c, vvTransFunc *tf, float, float);
    TUIVirvoWidget(TFKind myKind, vvTFWidget *w, vvTransFunc *tf);
    ~TUIVirvoWidget();

    virtual void setX(float x)
    {
        vvWidget->_pos[0] = x;
    }
    virtual float getX()
    {
        return vvWidget->_pos[0];
    }

    virtual void setY(float y)
    {
        vvWidget->_pos[1] = y;
    }
    virtual float getY()
    {
        return vvWidget->_pos[1];
    }

    void setAlpha(float a)
    {
        vvWidget->_opacity = a;
    }
    float getAlpha()
    {
        return vvWidget->_opacity;
    }

    bool hasOwnColor();
    void setOwnColor(bool f);

    virtual QColor getColor();
    virtual void setColor(QColor);
    void setColor(float r, float g, float b);
    void getColorFloat(float col[3]);

    void setXParam(int nParam, float value);
    float getXParam(int nParam);
    void setYParam(int nParam, float value);
    float getYParam(int nParam);

private:
    void setParam(int dim, int nParam, float value);
    float getParam(int dim, int nParam);

public:
    virtual void paint(QPainter &, int, bool) //painted by container (see GLCanvas)
    {
    }
    virtual bool contains(int x, int y, int w);

    virtual HandleType testHit(int x, int y);

    virtual void setActivated()
    {
        selectedHandle = HT_NONE;
    }
};
#endif //TUI_TFE_WIDGETS_H_INCLUDED
