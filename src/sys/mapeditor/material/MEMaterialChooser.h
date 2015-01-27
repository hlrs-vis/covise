/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_MATERIALCHOOSER_H
#define ME_MATERIALCHOOSER_H

#include <QWidget>
#include <QVector>
#include <QMultiMap>
#include <QDoubleSpinBox>

class QComboBox;
class QGroupBox;
class QCheckBox;
class QGridLayout;

class MEMaterialDisplay;
class MEMaterialSpinBox;

//================================================
class MEMaterialChooser : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEMaterialChooser(QWidget *parent = 0);
    void setMaterial(const QVector<float> &values);

signals:
    void materialChanged(const QVector<float> &data);

public slots:

    void loadMaterial(int index);
    void valueChanged(double d);

private slots:

    void selected();

private:
    int m_numUserMaterials, m_numMaterialGroups;

    MEMaterialDisplay *m_current, *m_edit;
    QComboBox *m_nameBox;
    QGroupBox *m_group;
    QGridLayout *m_grid;
    QStringList m_materialGroups;
    QVector<MEMaterialDisplay *> m_widgets;
    MEMaterialSpinBox *m_ambient[3], *m_diffuse[3], *m_specular[3], *m_emissive[3];
    MEMaterialSpinBox *m_shininess, *m_transparency;
    QMap<QString, QVector<float> > m_materials; // list contains values for one materialname
    QMultiMap<QString, QString> m_materialNameList; // list contains groupname & materialnames

    void readMaterials();
    void makeEditor();
    void updateSpinBoxValues(const QVector<float> &data);
    void updateGLWidget(const QVector<float> &data);
    void convertValue(const QString &name, const QString &var, const QString &def, QVector<float> &data);
};

//================================================
class MEMaterialSpinBox : public QDoubleSpinBox
//================================================
{
public:
    MEMaterialSpinBox(QWidget *parent)
        : QDoubleSpinBox(parent)
    {
        setRange(0, 1.0);
        setDecimals(3);
        setSingleStep(0.01);
    }

    void setValue(double v)
    {
        bool block = signalsBlocked();
        blockSignals(true);
        QDoubleSpinBox::setValue(v);
        blockSignals(block);
    }
};

#endif
