/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_CONTROLPARAMETER_H
#define ME_CONTROLPARAMETER_H

#include <QVector>
#include <QFrame>

class QVBoxLayout;
class QLabel;
class QPushButton;

class MENode;
class MEControlParameter;
class MEControlParameterLine;

//================================================
class MEControlParameter : public QFrame
//================================================
{
    Q_OBJECT

public:
    MEControlParameter(MENode *);
    ~MEControlParameter();

    int getLabelWidth()
    {
        return m_labelWidth;
    };
    void setNodeTitle(const QString &text);
    void removeParameter(MEControlParameterLine *);
    void insertParameter(MEControlParameterLine *);
    void setMasterState(bool);

    QWidget *getContainer()
    {
        return m_mainContent;
    };

    MENode *getNode()
    {
        return m_node;
    };

private:
    int m_labelWidth;
    bool m_fopen;

    MENode *m_node;

    QVBoxLayout *m_vlist;
    QWidget *m_mainContent;
    QPushButton *m_infoPB, *m_showPB, *m_execPB, *m_helpPB;
    QLabel *m_moduleTitle;

    QFrame *createHeader();
    QVector<MEControlParameterLine *> m_parameterList;

private slots:

    void showCB();

public slots:

    void recalculate(int);
    void bookCB();
};
#endif
