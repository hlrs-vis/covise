/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_CONTROLPARAMETERLINE_H
#define ME_CONTROLPARAMETERLINE_H

#include <QWidget>

class QHBoxLayout;
class QComboBox;
class QLabel;

class MEParameterPort;

//================================================
class MEControlParameterLine : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEControlParameterLine(QWidget *w, MEParameterPort *p);
    ~MEControlParameterLine();

    QLabel *getLabel()
    {
        return m_parameterName;
    };
    MEParameterPort *getPort()
    {
        return m_port;
    };
    void colorTextFrame(bool);

private:
    MEParameterPort *m_port;

    QLabel *m_parameterName;
    QHBoxLayout *m_boxLayout;
    QComboBox *cmap;

    void makeLayout();

signals:

    void recalculateSize(int);
};
#endif
