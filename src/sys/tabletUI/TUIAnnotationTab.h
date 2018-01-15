/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_ANNOTAION_TAB_H
#define CO_TUI_ANNOTAION_TAB_H

#include "TUITab.h"
#include "TUIApplication.h"
#include "TUISGBrowserTab.h"
#include "TUITextCheck.h"
#include <util/coTabletUIMessages.h>

#include <QObject>
#include <QListWidgetItem>

#include <QVariant>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QSlider>
#include <QSpacerItem>
#include <QTextEdit>
#include <QWidget>
#include <QFrame>

const int IDRole = Qt::UserRole;
const int TextRole = Qt::UserRole + 1;

class TUIAnnotationTab : public QObject, public TUITab
{
    Q_OBJECT

public:
    TUIAnnotationTab(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIAnnotationTab();

    virtual const char *getClassName();
    virtual void setValue(int type, covise::TokenBuffer &tb);

private:
    QFrame *frame;
    QGridLayout *gridLayout;
    QGroupBox *groupBoxCurrent;
    QWidget *layoutWidget;
    QGridLayout *gridLayout1;
    QSlider *sliderColor;
    QCheckBox *checkBoxShowHide;
    QLabel *labelScale;
    QPushButton *pushButtonDelete;
    QLabel *labelColor;
    QSlider *sliderScale;
    QPushButton *pushButtonOk;
    QSpacerItem *spacerItem;
    QTextEdit *textEdit;
    QGroupBox *groupBoxAll;
    QWidget *layoutWidget1;
    QGridLayout *gridLayout2;
    QPushButton *pushButtonDeleteAll;
    QSlider *sliderScaleAll;
    QLabel *labelColorAll;
    QSpacerItem *spacerItem1;
    QLabel *labelScaleAll;
    QCheckBox *checkBoxShowHideAll;
    QSlider *sliderColorAll;
    QPushButton *pushButtonNew;
    QSpacerItem *spacerItem2;
    QListWidget *listWidget;
    QListWidgetItem *newListItem;

private slots:

    void newAnnotation();
    void deleteAnnotation();
    void deleteAllAnnotations();
    void scaleAnnotation(int value);
    void scaleAllAnnotations(int value);
    void setAnnotationColor(int value);
    void setAllAnnotationColors(int value);
    void showOrHideAnnotation(int state);
    void showOrHideAllAnnotations(int state);
    void sendText();
    void itemClicked(QListWidgetItem *item);
};
#endif
