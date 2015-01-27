/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VALIDATEDQLINEEDIT
#define VALIDATEDQLINEEDIT

#include <QLineEdit>

class coEditorValidatedQLineEdit : public QLineEdit
{
    Q_OBJECT
public:
    coEditorValidatedQLineEdit(QWidget *parent = 0);
    bool isValid();

signals:
    void focusOut(); // TODO do i need that
    void notValid();

protected:
    void focusOutEvent(QFocusEvent *e);
    void keyPressEvent(QKeyEvent *e);

    QSize minimumSizeHint() const;

private:
    QString oldText;
};
#endif
