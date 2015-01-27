/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_COLORMAPPORT_H
#define ME_COLORMAPPORT_H

#include <QVector>
#include "ports/MEParameterPort.h"

class QStringList;
class QWidget;
class QComboBox;
class QDialog;
class QVBoxLayout;

class MENode;
class MEColorMap;
class MELineEdit;
class MEColorRGBTable;
class MEExtendedPart;

//================================================
class MEColorMapPort : public MEParameterPort
//================================================
{

    Q_OBJECT

public:
    MEColorMapPort(MENode *node, QGraphicsScene *scene, const QString &pportname, const QString &paramtype, const QString &description);
    MEColorMapPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEColorMapPort();

    int cmapSteps;
    int cmapStepsold;
    float *cmapRGBAX;
    float *cmapRGBAXold;
    int *values;

    void defineParam(QString value, int apptype);
    void modifyParam(QStringList list, int noOfValues, int istart);
    void modifyParameter(QString value);
    void sendParamMessage();
    void moduleParameterRequest();
    void makeLayout(layoutType type, QWidget *);
    void restoreParam();
    void storeParam();

#ifdef YAC
    void addPorts(covise::coRecvBuffer &);
    void setValues(covise::coRecvBuffer &);
#endif

    MEColorMap *getColorMap()
    {
        return m_colorMap;
    }

private slots:

    void folderCB();

public slots:

    void colorMapClosed();

private:
    bool m_fileOpen;

    MEColorMap *m_colorMap;
    MEExtendedPart *m_extendedPart[2];
    MEColorRGBTable *m_preview[2];

    QDialog *m_dialog;

    void switchExtendedPart();
    void changeFolderPixmap();
    void removeFromControlPanel();
    void addToControlPanel();
    void manageDialog();
    QString makeColorMapValues();
};
#endif
