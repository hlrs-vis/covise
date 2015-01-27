/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_PARAMLINE_H
#define ME_PARAMLINE_H

#include <QObject>

class QFrame;
class QPushButton;
class QWidget;

class MEParameterPort;
class MEParameterAppearance;

//================================================
class MEModuleParameterLine : public QObject
//================================================
{
    Q_OBJECT

public:
    MEModuleParameterLine(MEParameterPort *port, QFrame *frame, QWidget *widget);
    ~MEModuleParameterLine();

    void reset();
    void setEnabled(bool);
    void changeMappedPixmap(bool);
    void changeLightPixmap(bool);
    void colorTextFrame(bool);
    QPushButton *getLightButton()
    {
        return m_lightPB;
    };

private:
    MEParameterPort *m_port;
    MEParameterAppearance *m_appearanceTypes;

    QFrame *m_textFrame;
    QWidget *m_container, *m_secondLine;
    QPushButton *m_mappedPB, *m_lightPB;

private slots:

    void lightCB();
    void mappedCB();
};
#endif
