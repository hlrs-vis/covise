/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_SLIDERPORT_H
#define ME_SLIDERPORT_H

#include "ports/MEParameterPort.h"

class QSlider;
class QHBoxLayout;

class MELineEdit;

//================================================
class MESliderPort : public MEParameterPort
//================================================
{

    Q_OBJECT

public:
    MESliderPort(MENode *node, QGraphicsScene *scene, const QString &pportname, const QString &paramtype, const QString &description);
    MESliderPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MESliderPort();

protected:
    void removeFromControlPanel();
    void makePlayer(layoutType type, QWidget *w, const QString &value);
    void makeControlLine(layoutType type, QWidget *w, const QString &value);
    void makeModuleLine(layoutType type, QWidget *w, const QStringList &values);
    virtual QSlider *makeSlider(QWidget *parent) = 0;

    MELineEdit *m_textField;
    QSlider *m_slider[2];
    QVector<MELineEdit *> m_editList;
};
#endif
