/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_COLORPORT_H
#define ME_COLORPORT_H

#include <QVector>

#include "ports/MEParameterPort.h"

class QStringList;
class QWidget;
class QDialog;

class MELineEdit;
class MEExtendedPart;
class MEColorSelector;
class MEColorDisplay;
class MEColorChooser;

//================================================
class MEColorPort : public MEParameterPort
//================================================
{

    Q_OBJECT

public:
    MEColorPort(MENode *node, QGraphicsScene *scene, const QString &pportname, const QString &paramtype, const QString &description);
    MEColorPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEColorPort();

    void defineParam(QString value, int apptype);
    void modifyParameter(QString value);
    void modifyParam(QStringList list, int noOfValues, int istart);
    void sendParamMessage();
    void moduleParameterRequest();
    void makeLayout(layoutType type, QWidget *);
    void restoreParam();
    void storeParam();
    void colorMapClosed();

#ifdef YAC
    void addPorts(covise::coRecvBuffer &);
    void setValues(covise::coRecvBuffer &);
#endif

public slots:

    void showColor(const QColor &);

private slots:

    void textCB(const QString &);
    void folderCB();
    void okCB();
    void newColor(const QColor &m_color);

private:
    bool m_fileOpen;
    int m_currAlpha;
    float m_red, m_redold, m_green, m_greenold;
    float m_blue, m_blueold, m_alpha, m_alphaold;

    QColor m_color, m_currColor;
    MEColorDisplay *m_preview[2];
    QDialog *m_dialog;
    MEColorSelector *m_colorPicker[2];
    MEColorChooser *m_chooser;
    QVector<MELineEdit *> m_dataList[2];

    void changeFolderPixmap();
    void removeFromControlPanel();
    void addToControlPanel();
    void updateItems();
    void manageDialog();
};

#endif
