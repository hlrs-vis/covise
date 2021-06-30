/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef NONBLOCKINGDIALOGUE_H
#define NONBLOCKINGDIALOGUE_H

#include <QDialog>
#include <vector>
#include "export.h"
class QPushPutton;
class QLabel;

namespace Ui
{
    class dialog;
}

namespace covise
{

class QTUTIL_EXPORT NonBlockingDialogue : public QDialog
{
    Q_OBJECT
public:
    explicit NonBlockingDialogue(QWidget *parent = nullptr);
    void setInfo(const QString &text);
    void setQuestion(const QString &text);
    int addOption(const QString &text);
signals:
    void answer(int options);

private:
    const QWidget *m_parent = nullptr;
    Ui::dialog *m_ui;

protected:
    void showEvent(QShowEvent *event) override;

};
}

#endif // !NONBLOCKINGDIALOGUE_H