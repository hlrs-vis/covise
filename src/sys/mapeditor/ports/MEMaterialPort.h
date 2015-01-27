/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_MATERIALPORT_H
#define ME_MATERIALPORT_H

#include <QVector>
#include <QMap>

#include "ports/MEParameterPort.h"

class QStringList;

class MENode;
class MEMaterialDisplay;
class MEMaterialChooser;

//================================================
class MEMaterialPort : public MEParameterPort
//================================================
{

    Q_OBJECT

public:
    MEMaterialPort(MENode *node, QGraphicsScene *scene, const QString &pportname, const QString &paramtype, const QString &description);
    MEMaterialPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEMaterialPort();

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

public slots:

    void applyCB();
    void resetCB();
    void materialMapClosed();

private slots:

    void folderCB();
    void materialChanged(const QVector<float> &data);

private:
    bool m_fileOpen;

    QDialog *m_dialog;
    MEMaterialDisplay *m_preview[2];
    MEMaterialChooser *m_chooser;

    QString m_name, m_nameOld;
    QVector<float> m_values, m_valuesOld;

    void changeFolderPixmap();
    void removeFromControlPanel();
    void addToControlPanel();
    void updateItems();
    void manageDialog();
    void showMaterial();
};
#endif
