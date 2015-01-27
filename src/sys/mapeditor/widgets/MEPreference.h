/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_PREFERENCE_H
#define ME_PREFERENCE_H

#include <QFrame>

class QStringList;
class QCheckBox;
class QComboBox;

//================================================
class MEPreference : public QFrame
//================================================
{

    Q_OBJECT

public:
    MEPreference(QWidget *parent = 0);
    ~MEPreference();

    static MEPreference *instance();

    void update(QString, QString);

private:
    QComboBox *caching;
    QComboBox *registry;
    QCheckBox *execConn;
    QCheckBox *debugToggle;

    QStringList styles;

public slots:

    void debugPressed();
    void cachSelected(int);
    void regModeSelected(int);
    void execConnSelected();
};
#endif
