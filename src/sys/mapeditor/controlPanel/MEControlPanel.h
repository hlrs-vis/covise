/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_CONTROLPANEL_H
#define ME_CONTROLPANEL_H

#include <QVector>
#include <QFrame>

class QVBoxLayout;
class QLabel;
class MEControlParameter;

class MEControlPanel;

//================================================
class MEControlPanel : public QFrame
//================================================
{
    Q_OBJECT

public:
    static MEControlPanel *instance();

    MEControlPanel(QWidget *parent = 0);
    ~MEControlPanel();

    void addControlInfo(MEControlParameter *);
    void removeControlInfo(MEControlParameter *);
    void setMasterState(bool);
    QSize sizeHint() const;

private:
    QVBoxLayout *m_boxLayout;
    QVector<MEControlParameter *> m_controlList;
};
#endif
