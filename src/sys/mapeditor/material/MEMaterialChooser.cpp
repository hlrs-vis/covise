/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "config/coConfig.h"

#include <QHBoxLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QCheckBox>
#include <QDebug>

#include "MEMaterialChooser.h"
#include "MEMaterialDisplay.h"
#include "material.inc"

static int nValues = 14;
static QString userDef = "User Defined";

/*!
   \class MEMaterialChooser
   \brief This class provides a material chooser widget
*/

MEMaterialChooser::MEMaterialChooser(QWidget *parent)
    : QWidget(parent)
    , m_current(NULL)
    , m_edit(NULL)
{
    // get all materials
    readMaterials();

    // make main layout
    makeEditor();
}

#define addSpinBox(widget)                                                           \
    widget = new MEMaterialSpinBox(gb);                                              \
    connect(widget, SIGNAL(valueChanged(double)), this, SLOT(valueChanged(double))); \
    vbox->addWidget(widget, 0);

//!
//! make layout and widgets
//!
void MEMaterialChooser::makeEditor()
{
    // set main layout
    QHBoxLayout *hbox = new QHBoxLayout(this);

    // first column of hbox
    QWidget *w = new QWidget(this);
    QVBoxLayout *one = new QVBoxLayout(w);

    // add a combobox with maeterial m_group names
    QGroupBox *gb = new QGroupBox("Available material groups", w);
    QVBoxLayout *vbox = new QVBoxLayout(gb);

    // user defined materials
    m_nameBox = new QComboBox(gb);
    connect(m_nameBox, SIGNAL(activated(int)), this, SLOT(loadMaterial(int)));
    m_nameBox->addItems(m_materialGroups);
    vbox->addWidget(m_nameBox);
    one->addWidget(gb);

    // show content of one material group
    m_group = new QGroupBox(w);
    m_grid = new QGridLayout(m_group);
    loadMaterial(0);
    one->addWidget(m_group);
    one->addStretch(10);

    hbox->addWidget(w);

    // first row containing edit functions
    QWidget *container = new QWidget(this);
    QVBoxLayout *containerLayout = new QVBoxLayout(container);

    w = new QWidget(container);
    QHBoxLayout *two = new QHBoxLayout(w);

    gb = new QGroupBox("Preview", this);
    vbox = new QVBoxLayout(gb);
    m_edit = new MEMaterialDisplay(gb);
    vbox->addWidget(m_edit);
    two->addWidget(gb);

    gb = new QGroupBox("Shininess", this);
    vbox = new QVBoxLayout(gb);
    addSpinBox(m_shininess);
    vbox->addStretch(10);
    two->addWidget(gb);

    gb = new QGroupBox("Transparency", this);
    vbox = new QVBoxLayout(gb);
    addSpinBox(m_transparency);
    vbox->addStretch(10);
    two->addWidget(gb);
    containerLayout->addWidget(w);

    // second row containing edit functions
    w = new QWidget(container);
    QHBoxLayout *three = new QHBoxLayout(w);

    gb = new QGroupBox("Ambient", w);
    vbox = new QVBoxLayout(gb);
    addSpinBox(m_ambient[0]);
    addSpinBox(m_ambient[1]);
    addSpinBox(m_ambient[2]);
    three->addWidget(gb, 1);

    gb = new QGroupBox("Diffuse", w);
    vbox = new QVBoxLayout(gb);
    addSpinBox(m_diffuse[0]);
    addSpinBox(m_diffuse[1]);
    addSpinBox(m_diffuse[2]);
    three->addWidget(gb, 1);

    gb = new QGroupBox("Specular", this);
    vbox = new QVBoxLayout(gb);
    addSpinBox(m_specular[0]);
    addSpinBox(m_specular[1]);
    addSpinBox(m_specular[2]);
    three->addWidget(gb, 1);

    gb = new QGroupBox("Emissive", this);
    vbox = new QVBoxLayout(gb);
    addSpinBox(m_emissive[0]);
    addSpinBox(m_emissive[1]);
    addSpinBox(m_emissive[2]);
    three->addWidget(gb, 1);
    containerLayout->addWidget(w);
    hbox->addWidget(container);
}

//!
//! show all materials within a group
//!
void MEMaterialChooser::loadMaterial(int)
{
    // show the materials for a given m_group
    QString text = "Materials for group: " + m_nameBox->currentText();
    m_group->setTitle(text);
    QStringList nameList(m_materialNameList.values(m_nameBox->currentText()));
    nameList.sort();

    // clear old content
    // don't delete widgets if  size is the same
    int size = nameList.size();
    int oldsize = m_widgets.size();

    if (oldsize > size)
    {
        for (int i = size; i < oldsize; ++i)
            delete m_widgets[i];
        m_widgets.resize(size);
    }

    for (int i = oldsize; i < size; ++i)
    {
        MEMaterialDisplay *disp = new MEMaterialDisplay(m_group);
        connect(disp, SIGNAL(clicked()), this, SLOT(selected()));
        m_widgets.append(disp);
        m_grid->addWidget(disp, i / 6, i % 6);
    }

    // fill with new content
    int i = 0;
    foreach (QString text, nameList)
    {
        QVector<float> data = m_materials.value(text);
        MEMaterialDisplay *disp = m_widgets[i];
        disp->setValues(data);
        disp->setToolTip(text);
        ++i;
    }
}

//!-------------------------------------------------------------------------
//! change inventor material values (string -> float)
//!-------------------------------------------------------------------------
void MEMaterialChooser::convertValue(const QString &name, const QString &value, const QString &def,
                                     QVector<float> &data)
{
    covise::coConfig *config = covise::coConfig::getInstance();
    QString text = config->getString(value, name, def);
    QStringList list = text.split(' ', QString::SkipEmptyParts);
    data.append(list[0].toFloat());
    data.append(list[1].toFloat());
    data.append(list[2].toFloat());
}

//!-------------------------------------------------------------------------
//! read all available materials
//!-------------------------------------------------------------------------
void MEMaterialChooser::readMaterials()
{
    // read materials from material.inc (Inventor materials)
    m_numMaterialGroups = num_mat_groups;

    int ie = 0;
    for (int i = 0; i < m_numMaterialGroups; i++)
    {
        m_materialGroups.append(mat_groups[i]);
        int num = num_materials_gr[i];
        for (int k = 0; k < num; k++)
        {
            QVector<float> data;
            QString matname = mat_values[ie][0];
            for (int j = 0; j < nValues; j++)
            {
                QString val = mat_values[ie][j + 1];
                data.append(val.toFloat());
            }
            m_materials.insert(matname, data);
            m_materialNameList.insert(mat_groups[i], matname);
            ie++;
        }
    }
    m_materialGroups.sort();

    // read the name of all materials in config files (user defined materials)
    covise::coConfig *config = covise::coConfig::getInstance();
    QString scope = "Module.Material";
    QStringList list;
    list = config->getVariableList(scope);

    m_numUserMaterials = list.size();
    if (m_numUserMaterials != 0)
        m_materialGroups.append(userDef);

    // read the values for each materials
    for (int k = 0; k < m_numUserMaterials; k++)
    {
        QVector<float> data;
        QString name = scope + "." + list[k];

        convertValue(name, "ambient", "0.7 0.7 0.7", data);
        convertValue(name, "diffuse", "0.1 0.5 0.8", data);
        convertValue(name, "specular", "1.0 1.0 1.0", data);
        convertValue(name, "emissive", "0.3 0.2 0.2", data);
        data.append(config->getFloat("shininess", name, 0.2f));
        data.append(config->getFloat("transparency", name, 1.0f));
        QString matname = config->getString("name", name, "Standard");
        m_materials.insert(matname, data);
        m_materialNameList.insert(userDef, matname);
    }
}

//!-------------------------------------------------------------------------
//! update spinboxes with new values
//!-------------------------------------------------------------------------
void MEMaterialChooser::updateSpinBoxValues(const QVector<float> &data)
{
    m_ambient[0]->setValue(data[0]);
    m_ambient[1]->setValue(data[1]);
    m_ambient[2]->setValue(data[2]);
    m_diffuse[0]->setValue(data[3]);
    m_diffuse[1]->setValue(data[4]);
    m_diffuse[2]->setValue(data[5]);
    m_specular[0]->setValue(data[6]);
    m_specular[1]->setValue(data[7]);
    m_specular[2]->setValue(data[8]);
    m_emissive[0]->setValue(data[9]);
    m_emissive[1]->setValue(data[10]);
    m_emissive[2]->setValue(data[11]);
    m_shininess->setValue(data[12]);
    m_transparency->setValue(data[13]);
}

//!-------------------------------------------------------------------------
//! update the OpenGL widget with new values
//!-------------------------------------------------------------------------
void MEMaterialChooser::updateGLWidget(const QVector<float> &data)
{
    if (m_edit)
        m_edit->setValues(data);
}

//!-------------------------------------------------------------------------
//! port send a new material (modifyParam)
//!-------------------------------------------------------------------------
void MEMaterialChooser::setMaterial(const QVector<float> &data)
{
    if (data.size() > 12)
    {
        updateSpinBoxValues(data);
        updateGLWidget(data);
    }
}

//!-------------------------------------------------------------------------
//! callback: a new material was selected
//!-------------------------------------------------------------------------
void MEMaterialChooser::selected()
{
    // get the object that sent the signal
    const QObject *obj = sender();

    if (m_widgets.contains(m_current))
        m_current->setSelected(false);
    m_current = (MEMaterialDisplay *)obj;

    // get new values, update spinboxes & OpenGL widget
    const QVector<float> data = m_current->getValues();
    setMaterial(data);

    // inform material parameter port
    materialChanged(data);
}

//!-------------------------------------------------------------------------
//! callback: value of a spinbox was changed
//!-------------------------------------------------------------------------
void MEMaterialChooser::valueChanged(double)
{
    // get values
    QVector<float> data;
    data.append(m_ambient[0]->value());
    data.append(m_ambient[1]->value());
    data.append(m_ambient[2]->value());
    data.append(m_diffuse[0]->value());
    data.append(m_diffuse[1]->value());
    data.append(m_diffuse[2]->value());
    data.append(m_specular[0]->value());
    data.append(m_specular[1]->value());
    data.append(m_specular[2]->value());
    data.append(m_emissive[0]->value());
    data.append(m_emissive[1]->value());
    data.append(m_emissive[2]->value());
    data.append(m_shininess->value());
    data.append(m_transparency->value());

    // update OpenGl widget & inform material parameter port
    updateGLWidget(data);
    materialChanged(data);
}
