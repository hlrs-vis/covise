/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_FLOATSLIDERPORT_H
#define ME_FLOATSLIDERPORT_H

#include "ports/MESliderPort.h"

class QStringList;

//================================================
class MEFloatSliderPort : public MESliderPort
//================================================
{

    Q_OBJECT

public:
    MEFloatSliderPort(MENode *node, QGraphicsScene *scene, const QString &pportname, const QString &paramtype, const QString &description);
    MEFloatSliderPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEFloatSliderPort();

#ifdef YAC
    void setValues(covise::coRecvBuffer &);
#endif

    void restoreParam();
    void storeParam();
    void sendParamMessage();
    void moduleParameterRequest();
    void defineParam(QString value, int apptype);
    void modifyParam(QStringList list, int noOfValues, int istart);
    void modifyParameter(QString value);
    void makeLayout(layoutType type, QWidget *w);

private slots:

    void textCB(const QString &);
    void text2CB(const QString &);
    void boundaryCB();
    void slider1CB(int);
    void slider2CB();

private:
    float m_value, m_min, m_max, m_step;
    float m_valueold, m_minold, m_maxold, m_stepold;

    int fToI(float fvalue);
    float iToF(int ivalue);
    void plusNewValue();
    void minusNewValue();
    QSlider *makeSlider(QWidget *parent);
};
#endif
