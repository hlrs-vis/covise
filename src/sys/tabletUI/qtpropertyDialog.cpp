/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
 **
 */
#include "qtpropertyDialog.h"
#include "color/MEColorChooser.h"
#include <QDebug>
#include <QQuaternion>
#include <QIcon>
#include <QPainter>
#include <QSlider>
#include <cmath>
#ifndef ANDROID_TUI
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#else
#define GL_POINTS 0x0000
#define GL_LINES 0x0001
#define GL_LINE_LOOP 0x0002
#define GL_LINE_STRIP 0x0003
#define GL_TRIANGLES 0x0004
#define GL_TRIANGLE_STRIP 0x0005
#define GL_TRIANGLE_FAN 0x0006
#define GL_QUADS 0x0007
#define GL_QUAD_STRIP 0x0008
#define GL_POLYGON 0x0009
#endif

   void getEulerAngles(QQuaternion quat, float *pitch, float *yaw, float *roll)
   {
       Q_ASSERT(pitch && yaw && roll);
    float xp,yp,zp,wp;
    xp = quat.x();
    yp = quat.y();
    zp = quat.z();
    wp = quat.scalar();

       // Algorithm from:
       // http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q37
   
       float xx = xp * xp;
       float xy = xp * yp;
       float xz = xp * zp;
       float xw = xp * wp;
       float yy = yp * yp;
       float yz = yp * zp;
       float yw = yp * wp;
       float zz = zp * zp;
       float zw = zp * wp;
   
       const float lengthSquared = xx + yy + zz + wp * wp;
       if (!qFuzzyIsNull(lengthSquared - 1.0f) && !qFuzzyIsNull(lengthSquared)) {
           xx /= lengthSquared;
           xy /= lengthSquared; // same as (xp / length) * (yp / length)
           xz /= lengthSquared;
           xw /= lengthSquared;
           yy /= lengthSquared;
           yz /= lengthSquared;
           yw /= lengthSquared;
           zz /= lengthSquared;
           zw /= lengthSquared;
       }
   
       *pitch = std::asin(-2.0f * (yz - xw));
       if (*pitch < M_PI_2) {
           if (*pitch > -M_PI_2) {
               *yaw = std::atan2(2.0f * (xz + yw), 1.0f - 2.0f * (xx + yy));
               *roll = std::atan2(2.0f * (xy + zw), 1.0f - 2.0f * (xx + zz));
           } else {
               // not a unique solution
               *roll = 0.0f;
               *yaw = -std::atan2(-2.0f * (xy - zw), 1.0f - 2.0f * (yy + zz));
           }
       } else {
           // not a unique solution
           *roll = 0.0f;
           *yaw = std::atan2(-2.0f * (xy - zw), 1.0f - 2.0f * (yy + zz));
       }
   
       *pitch = (*pitch)*180.0/M_PI;
       *yaw = (*yaw)*180.0/M_PI;
       *roll = (*roll)*180.0/M_PI;
   }
QQuaternion fromRotationMatrix(const QMatrix3x3 &rot3x3)
{
    // Algorithm from:
    // http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q55

    float scalar;
    float axis[3];

    const float trace = rot3x3(0, 0) + rot3x3(1, 1) + rot3x3(2, 2);
    if (trace > 0.00000001f) {
        const float s = 2.0f * std::sqrt(trace + 1.0f);
        scalar = 0.25f * s;
        axis[0] = (rot3x3(2, 1) - rot3x3(1, 2)) / s;
        axis[1] = (rot3x3(0, 2) - rot3x3(2, 0)) / s;
        axis[2] = (rot3x3(1, 0) - rot3x3(0, 1)) / s;
    } else {
        static int s_next[3] = { 1, 2, 0 };
        int i = 0;
        if (rot3x3(1, 1) > rot3x3(0, 0))
            i = 1;
        if (rot3x3(2, 2) > rot3x3(i, i))
            i = 2;
        int j = s_next[i];
        int k = s_next[j];

        const float s = 2.0f * std::sqrt(rot3x3(i, i) - rot3x3(j, j) - rot3x3(k, k) + 1.0f);
        axis[i] = 0.25f * s;
        scalar = (rot3x3(k, j) - rot3x3(j, k)) / s;
        axis[j] = (rot3x3(j, i) + rot3x3(i, j)) / s;
        axis[k] = (rot3x3(k, i) + rot3x3(i, k)) / s;
    }

    return QQuaternion(scalar, axis[0], axis[1], axis[2]);
}

 QQuaternion fromEulerAngles(float pitch, float yaw, float roll)
 {
     // Algorithm from:
     // http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q60
 
     pitch = (pitch)/180.0*M_PI;
     yaw = (yaw)/180.0*M_PI;
     roll = (roll)/180.0*M_PI;
 
     pitch *= 0.5f;
     yaw *= 0.5f;
     roll *= 0.5f;
 
     const float c1 = std::cos(yaw);
     const float s1 = std::sin(yaw);
     const float c2 = std::cos(roll);
     const float s2 = std::sin(roll);
     const float c3 = std::cos(pitch);
     const float s3 = std::sin(pitch);
     const float c1c2 = c1 * c2;
     const float s1s2 = s1 * s2;
 
     const float w = c1c2 * c3 + s1s2 * s3;
     const float x = c1c2 * s3 + s1s2 * c3;
     const float y = s1 * c2 * c3 - c1 * s2 * s3;
     const float z = c1 * s2 * c3 - s1 * c2 * s3;
 
     return QQuaternion(w, x, y, z);
 }
#define MY_GL_LINES_ADJACENCY_EXT 0x000A
#define MY_GL_LINE_STRIP_ADJACENCY_EXT 0x000B
#define MY_GL_TRIANGLES_ADJACENCY_EXT 0x000C
#define MY_GL_TRIANGLE_STRIP_ADJACENCY_EXT 0x000D
PropertyDialog::PropertyDialog(TUISGBrowserTab *tab, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Properties Dialog"));
    
    osgToVrml.rotate(90,1,0,0);
    vrmlToOsg.rotate(-90,1,0,0);

    QGridLayout *mainLayout = new QGridLayout();
    doRemove = false;
    layout = new QGridLayout();

    QLabel *nameL = new QLabel("Name:");
    QLabel *typeL = new QLabel("Type:");
    _name = new QLabel();
    _type = new QLabel();

    numL = new QLabel("numChildren:");
    _num = new QLabel();

    R_line = new QRadioButton();
    R_all = new QRadioButton();
    R_line->setText("Show children:");
    R_all->setText("Show all children");
    R_group = new QButtonGroup();
    R_group->addButton(R_line);
    R_group->addButton(R_all);

    childrenEdit = new QLineEdit();
    childrenEdit->setToolTip(" from 1 to numChildren \n e.g.: 1;3;5-12 \n 0 to hide all");

    modeCB = new QCheckBox("Render depth only");
    transCB = new QCheckBox("Transparency");

    layout->addWidget(nameL, 0, 0, Qt::AlignLeft);
    layout->addWidget(typeL, 1, 0, Qt::AlignLeft);
    layout->addWidget(_name, 0, 1, Qt::AlignLeft);
    layout->addWidget(_type, 1, 1, Qt::AlignLeft);
    layout->addWidget(numL, 2, 0, Qt::AlignLeft);
    layout->addWidget(_num, 2, 1, Qt::AlignLeft);
    layout->addWidget(R_line, 3, 0, Qt::AlignLeft);
    layout->addWidget(childrenEdit, 3, 1, Qt::AlignLeft);
    layout->addWidget(R_all, 4, 0, Qt::AlignLeft);
    layout->addWidget(modeCB, 5, 0, Qt::AlignLeft);
    layout->addWidget(transCB, 6, 0, Qt::AlignLeft);

    QGridLayout *tabLayout = new QGridLayout();
    QTabWidget *_tab = new QTabWidget();
    connect(_tab, SIGNAL(currentChanged(int)), this, SLOT(onTabPressed(int)));

    tabLayout->addWidget(_tab, 0, 0, Qt::AlignCenter);

    //page1 ///////////////////////////////////////////////////////////////////////////////////////////

    diffuseP = new QPixmap(20, 20);
    diffuseP->fill(QColor(0, 0, 0));
    diffuseL = new QLabel();
    diffuseL->setPixmap(*diffuseP);

    ambientP = new QPixmap(20, 20);
    ambientP->fill(QColor(0, 0, 0));
    ambientL = new QLabel();
    ambientL->setPixmap(*ambientP);

    specularP = new QPixmap(20, 20);
    specularP->fill(QColor(0, 0, 0));
    specularL = new QLabel();
    specularL->setPixmap(*specularP);

    emissiveP = new QPixmap(20, 20);
    emissiveP->fill(QColor(0, 0, 0));
    emissiveL = new QLabel();
    emissiveL->setPixmap(*emissiveP);

    QWidget *page1 = new QWidget();
    QGridLayout *page1Layout = new QGridLayout();
    diffuseCB = new QCheckBox("Diffuse");
    diffuseCB->setCheckState(Qt::Checked);
    specularCB = new QCheckBox("Specular");
    ambientCB = new QCheckBox("Ambient");
    ambientCB->setCheckState(Qt::Checked);
    emissiveCB = new QCheckBox("Emissive");
    removeMat = new QPushButton("Remove Material");
    colorchooser = new MEColorChooser();

    page1Layout->addWidget(diffuseCB, 0, 0, Qt::AlignLeft);
    page1Layout->addWidget(diffuseL, 0, 1, Qt::AlignLeft);
    page1Layout->addWidget(ambientCB, 1, 0, Qt::AlignLeft);
    page1Layout->addWidget(ambientL, 1, 1, Qt::AlignLeft);
    page1Layout->addWidget(specularCB, 0, 2, Qt::AlignLeft);
    page1Layout->addWidget(specularL, 0, 3, Qt::AlignLeft);
    page1Layout->addWidget(emissiveCB, 1, 2, Qt::AlignLeft);
    page1Layout->addWidget(emissiveL, 1, 3, Qt::AlignLeft);
    page1Layout->addWidget(removeMat, 2, 0, Qt::AlignLeft);

    QVBoxLayout *page1vlayout = new QVBoxLayout();

    page1vlayout->addLayout(page1Layout);
    page1vlayout->addWidget(colorchooser);
    connect(colorchooser, SIGNAL(colorChanged(const QColor &)), this, SLOT(onColorChanged(const QColor &)));

    page1->setLayout(page1vlayout);

    //page2 ///////////////////////////////////////////////////////////////////////////////////////////
    QWidget *page2 = new QWidget();
    QGridLayout *page2Layout = new QGridLayout();

    QGridLayout *listLayout = new QGridLayout();
    listWidget = new QListWidget;
    listWidget->setViewMode(QListView::IconMode);
    listWidget->setIconSize(QSize(96, 96));
    listWidget->setMovement(QListView::Static);
    listWidget->setMaximumWidth(150);
    listWidget->setSpacing(10);
    listWidget->setCurrentRow(0);

    QHBoxLayout *texBtnlayout = new QHBoxLayout();
    QPushButton *texLoadBtn = new QPushButton();
    texLoadBtn->setText("Load");
    texUpdateBtn = new QPushButton();
    texUpdateBtn->setText("Update");
    texBtnlayout->addWidget(texLoadBtn);
    texBtnlayout->addWidget(texUpdateBtn);

    connect(listWidget, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(updateView(QListWidgetItem *)));
    connect(texUpdateBtn, SIGNAL(clicked()), tab, SLOT(updateTextures()));
    connect(texLoadBtn, SIGNAL(clicked()), tab, SLOT(loadTexture()));

    listLayout->addWidget(listWidget, 0, 0, Qt::AlignHCenter);
    listLayout->addLayout(texBtnlayout, 1, 0, Qt::AlignHCenter);

    QGridLayout *menu = new QGridLayout();
    view = new QLabel();
    view->setFixedSize(96, 96);
    view->setAlignment(Qt::AlignCenter);
    menu->addWidget(view, 0, 0, Qt::AlignCenter);

    QGridLayout *texbtn = new QGridLayout();
    QLabel *texNum = new QLabel("Texture Number:");
    QLabel *texEnvMode = new QLabel("TexEnv Mode:");
    QLabel *texGenMode = new QLabel("TexGen Mode:");

    textureNumberSpin = new QSpinBox();
    textureNumberSpin->setObjectName("textureNumberSpin");
    textureNumberSpin->setRange(0, 20);
    textureNumberSpin->setValue(0);
    textureNumberSpin->setWrapping(true);
    connect(textureNumberSpin, SIGNAL(valueChanged(int)), tab, SLOT(changeTexMode(int)));

    textureModeComboBox = new QComboBox();
    textureModeComboBox->setEditable(false);
    textureModeComboBox->addItem("DECAL");
    textureModeComboBox->addItem("MODULATE");
    textureModeComboBox->addItem("BLEND");
    textureModeComboBox->addItem("REPLACE");
    textureModeComboBox->addItem("ADD");
    textureModeComboBox->setCurrentIndex(1);

    textureTexGenModeComboBox = new QComboBox();
    textureTexGenModeComboBox->setEditable(false);
    textureTexGenModeComboBox->addItem("OFF");
    textureTexGenModeComboBox->addItem("ObjectLinear");
    textureTexGenModeComboBox->addItem("EyeLinear");
    textureTexGenModeComboBox->addItem("SphereMap");
    textureTexGenModeComboBox->addItem("NormalMap");
    textureTexGenModeComboBox->addItem("ReflectionMap");
    textureTexGenModeComboBox->setCurrentIndex(0);

    texApplyButton = new QPushButton();
    texApplyButton->setText("Add");
    connect(texApplyButton, SIGNAL(clicked()), tab, SLOT(sendChangeTextureRequest()));
    //texApplyButton->setEnabled(false);

    texRemoveBtn = new QPushButton();
    texRemoveBtn->setText("Remove");
    connect(texRemoveBtn, SIGNAL(clicked()), tab, SLOT(handleRemoveTex()));
    //texRemoveBtn->setEnabled(false);

    texbtn->addWidget(texNum, 0, 0, Qt::AlignLeft);
    texbtn->addWidget(texEnvMode, 1, 0, Qt::AlignLeft);
    texbtn->addWidget(texGenMode, 2, 0, Qt::AlignLeft);

    texbtn->addWidget(textureNumberSpin, 0, 1, Qt::AlignLeft);
    texbtn->addWidget(textureModeComboBox, 1, 1, Qt::AlignLeft);
    texbtn->addWidget(textureTexGenModeComboBox, 2, 1, Qt::AlignLeft);

    QHBoxLayout *texBtnL = new QHBoxLayout();
    texBtnL->addWidget(texApplyButton);
    texBtnL->addWidget(texRemoveBtn);
    texbtn->addLayout(texBtnL, 4, 1, Qt::AlignRight);

    menu->addLayout(texbtn, 1, 0, Qt::AlignLeft);
    page2Layout->addLayout(listLayout, 0, 0, Qt::AlignLeft);
    page2Layout->addLayout(menu, 0, 1, Qt::AlignLeft);
    page2->setLayout(page2Layout);

    //page3 ///////////////////////////////////////////////////////////////////////////////////////////
    page3 = new QWidget();
    QGridLayout *page3Layout = new QGridLayout();

    QGridLayout *shaderListL = new QGridLayout();
    shaderListWidget = new QListWidget;
    shaderListWidget->setMovement(QListView::Static);
    shaderListWidget->setMaximumWidth(100);
    /*shaderGetBtn = new QPushButton();
   shaderGetBtn->setText("Get all");
   shaderGetBtn->setFixedWidth(100);*/
    QAction *updateAct = new QAction("Update", this);
    connect(updateAct, SIGNAL(triggered()), tab, SLOT(handleGetShader()));
    shaderListWidget->setContextMenuPolicy(Qt::ActionsContextMenu);
    shaderListWidget->addAction(updateAct);

    shaderSetBtn = new QPushButton();
    shaderSetBtn->setText("Set Shader");
    shaderSetBtn->setFixedWidth(100);

    shaderRemoveBtn = new QPushButton();
    shaderRemoveBtn->setText("Remove Shader");
    shaderRemoveBtn->setFixedWidth(100);

    shaderStoreBtn = new QPushButton();
    shaderStoreBtn->setText("Store Shader");
    shaderStoreBtn->setFixedWidth(100);
    /*
      shaderEditBtn = new QPushButton();
      shaderEditBtn->setText("Edit Shader >>");
      shaderEditBtn->setCheckable(true);
      shaderEditBtn->setFixedWidth(100);
   */
    //connect(shaderGetBtn, SIGNAL(clicked()), tab, SLOT(handleGetShader()));
    connect(shaderSetBtn, SIGNAL(clicked()), tab, SLOT(handleSetShader()));
    connect(shaderRemoveBtn, SIGNAL(clicked()), tab, SLOT(handleRemoveShader()));
    connect(shaderStoreBtn, SIGNAL(clicked()), tab, SLOT(handleStoreShader()));
    //connect(shaderEditBtn, SIGNAL(toggled(bool)), this, SLOT(handleEditShader(bool)));

    connect(shaderListWidget, SIGNAL(itemClicked(QListWidgetItem *)), tab, SLOT(handleShaderList(QListWidgetItem *)));
    shaderListL->addWidget(shaderListWidget, 0, 0, Qt::AlignLeft);
    //shaderListL->addWidget(shaderGetBtn, 1,0, Qt::AlignLeft);
    shaderListL->addWidget(shaderSetBtn, 2, 0, Qt::AlignLeft);
    shaderListL->addWidget(shaderRemoveBtn, 3, 0, Qt::AlignLeft);
    shaderListL->addWidget(shaderStoreBtn, 4, 0, Qt::AlignLeft);

    QGridLayout *shaderTabL = new QGridLayout();
    shaderTab = new QTabWidget();

    QWidget *page31 = new QWidget();
    QGridLayout *page31Layout = new QGridLayout();

    QGridLayout *uniformListL = new QGridLayout();
    uniformListWidget = new QListWidget;
    uniformListWidget->setMovement(QListView::Static);
    uniformListWidget->setMaximumWidth(100);
    connect(uniformListWidget, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(handleUniformList(QListWidgetItem *)));
    uniformListL->addWidget(uniformListWidget, 0, 0, Qt::AlignLeft);

    QGridLayout *uniformMenuL = new QGridLayout();

    VecEdit1 = new QLineEdit();
    VecEdit1->setMaximumWidth(120);
    connect(VecEdit1, SIGNAL(returnPressed()), this, SLOT(handleUniformVec()));
    VecEdit2 = new QLineEdit();
    VecEdit2->setMaximumWidth(120);
    connect(VecEdit2, SIGNAL(returnPressed()), this, SLOT(handleUniformVec()));
    VecEdit3 = new QLineEdit();
    VecEdit3->setMaximumWidth(120);
    connect(VecEdit3, SIGNAL(returnPressed()), this, SLOT(handleUniformVec()));
    VecEdit4 = new QLineEdit();
    VecEdit4->setMaximumWidth(120);
    connect(VecEdit4, SIGNAL(returnPressed()), this, SLOT(handleUniformVec()));

    VecEdit1->hide();
    VecEdit2->hide();
    VecEdit3->hide();
    VecEdit4->hide();

    uniformColorBtn = new QPushButton();
    uniformColorBtn->hide();
    connect(uniformColorBtn, SIGNAL(pressed()), this, SLOT(handleUniformVec4()));

    editLabel = new QLabel("Texture Number");
    editLabel->hide();
    uniformEdit = new QLineEdit();
    uniformEdit->setMaximumWidth(120);
    connect(uniformEdit, SIGNAL(returnPressed()), this, SLOT(handleSetUniformL()));
    uniformSlider = new QSlider();
    uniformSlider->setOrientation(Qt::Horizontal);
    uniformSlider->setMaximumWidth(120);
    connect(uniformSlider, SIGNAL(valueChanged(int)), this, SLOT(handleUniformInt(int)));
    uniformSlider->hide();

    texFileEditLabel = new QLabel("Default Texture");
    texFileEditLabel->hide();
    uniformTexFileEdit = new QLineEdit();
    uniformTexFileEdit->setMaximumWidth(250);
    uniformTexFileEdit->setWindowTitle("default texture");
    connect(uniformTexFileEdit, SIGNAL(returnPressed()), this, SLOT(handleUniformText()));
    uniformTexFileEdit->hide();
    reloadLabel = new QLabel("Please remove shader and set Shader again");
    reloadLabel->hide();
    //uniformAddBtn = new QPushButton();
    //uniformAddBtn->setText("Set value");
    //connect(uniformAddBtn, SIGNAL(clicked()), tab, SLOT(handleSetUniform()));

    uniformMenuL->addWidget(VecEdit1, 0, 0, Qt::AlignLeft);
    uniformMenuL->addWidget(VecEdit2, 1, 0, Qt::AlignLeft);
    uniformMenuL->addWidget(VecEdit3, 2, 0, Qt::AlignLeft);
    uniformMenuL->addWidget(VecEdit4, 3, 0, Qt::AlignLeft);
    uniformMenuL->addWidget(uniformColorBtn, 4, 0, Qt::AlignLeft);
    uniformMenuL->addWidget(editLabel, 5, 0, Qt::AlignLeft);
    uniformMenuL->addWidget(uniformEdit, 5, 1, Qt::AlignLeft);
    uniformMenuL->addWidget(uniformSlider, 6, 1, Qt::AlignLeft);
    uniformMenuL->addWidget(texFileEditLabel, 7, 0, Qt::AlignLeft);
    uniformMenuL->addWidget(uniformTexFileEdit, 7, 1, Qt::AlignLeft);
    uniformMenuL->addWidget(reloadLabel, 8, 0, Qt::AlignLeft);
    //uniformMenuL->addWidget(uniformAddBtn, 7,0, Qt::AlignRight);

    page31Layout->addLayout(uniformListL, 0, 0, Qt::AlignLeft);
    page31Layout->addLayout(uniformMenuL, 0, 1, Qt::AlignLeft);
    page31->setLayout(page31Layout);

    QWidget *page32 = new QWidget();
    QVBoxLayout *page32Layout = new QVBoxLayout();
    QHBoxLayout *vertexBtnLayout = new QHBoxLayout();

    vertexEdit = new QTextEdit();
    vertexEdit->setWordWrapMode(QTextOption::NoWrap);
    vertexBtn = new QPushButton();
    vertexBtn->setText("Set source");
    connect(vertexBtn, SIGNAL(clicked()), tab, SLOT(handleSetVertex()));
    vertexBtnLayout->addStretch();
    vertexBtnLayout->addWidget(vertexBtn);
    page32Layout->addWidget(vertexEdit);
    page32Layout->addLayout(vertexBtnLayout);

    page32->setLayout(page32Layout);

    QWidget *page35 = new QWidget();
    QVBoxLayout *page35Layout = new QVBoxLayout();
    QHBoxLayout *tessControlBtnLayout = new QHBoxLayout();

    tessControlEdit = new QTextEdit();
    tessControlEdit->setWordWrapMode(QTextOption::NoWrap);
    tessControlBtn = new QPushButton();
    tessControlBtn->setText("Set source");
    connect(tessControlBtn, SIGNAL(clicked()), tab, SLOT(handleSetTessControl()));
    tessControlBtnLayout->addStretch();
    tessControlBtnLayout->addWidget(tessControlBtn);
    page35Layout->addWidget(tessControlEdit);
    page35Layout->addLayout(tessControlBtnLayout);

    page35->setLayout(page35Layout);

    QWidget *page36 = new QWidget();
    QVBoxLayout *page36Layout = new QVBoxLayout();
    QHBoxLayout *tessEvalBtnLayout = new QHBoxLayout();

    tessEvalEdit = new QTextEdit();
    tessEvalEdit->setWordWrapMode(QTextOption::NoWrap);
    tessEvalBtn = new QPushButton();
    tessEvalBtn->setText("Set source");
    connect(tessEvalBtn, SIGNAL(clicked()), tab, SLOT(handleSetTessEval()));
    tessEvalBtnLayout->addStretch();
    tessEvalBtnLayout->addWidget(tessEvalBtn);
    page36Layout->addWidget(tessEvalEdit);
    page36Layout->addLayout(tessEvalBtnLayout);

    page36->setLayout(page36Layout);

    QWidget *page33 = new QWidget();
    QVBoxLayout *page33Layout = new QVBoxLayout();
    QHBoxLayout *fragmentBtnLayout = new QHBoxLayout();

    fragmentEdit = new QTextEdit();
    fragmentEdit->setWordWrapMode(QTextOption::NoWrap);
    fragmentBtn = new QPushButton();
    fragmentBtn->setText("Set source");
    connect(fragmentBtn, SIGNAL(clicked()), tab, SLOT(handleSetFragment()));
    fragmentBtnLayout->addStretch();
    fragmentBtnLayout->addWidget(fragmentBtn);
    page33Layout->addWidget(fragmentEdit);
    page33Layout->addLayout(fragmentBtnLayout);

    page33->setLayout(page33Layout);

    QWidget *page34 = new QWidget();
    QVBoxLayout *page34Layout = new QVBoxLayout();

    geometryEdit = new QTextEdit();
    geometryEdit->setWordWrapMode(QTextOption::NoWrap);
    geometryEdit->setMinimumWidth(200);
    geometryBtn = new QPushButton();
    geometryBtn->setText("Set source");
    connect(geometryBtn, SIGNAL(clicked()), tab, SLOT(handleSetGeometry()));

    geometryNumVertSpin = new QSpinBox();
    geometryNumVertSpin->setObjectName("textureNumberSpin");
    geometryNumVertSpin->setRange(1, 8000);
    geometryNumVertSpin->setValue(3);
    geometryNumVertSpin->setWrapping(true);
    connect(geometryNumVertSpin, SIGNAL(valueChanged(int)), tab, SLOT(handleNumVertChanged(int)));

    geometryInputModeComboBox = new QComboBox();
    geometryInputModeComboBox->setEditable(false);
    geometryInputModeComboBox->addItem("POINTS");
    geometryInputModeComboBox->addItem("LINES");
    geometryInputModeComboBox->addItem("LINES_ADJACENCY_EXT");
    geometryInputModeComboBox->addItem("TRIANGLES_ADJACENCY_EXT");
    geometryInputModeComboBox->addItem("TRIANGLES");
    geometryInputModeComboBox->setCurrentIndex(0);
    connect(geometryInputModeComboBox, SIGNAL(currentIndexChanged(int)), tab, SLOT(handleInputTypeChanged(int)));

    geometryOutputModeComboBox = new QComboBox();
    geometryOutputModeComboBox->setEditable(false);
    geometryOutputModeComboBox->addItem("POINTS");
    geometryOutputModeComboBox->addItem("LINE_STRIP");
    geometryOutputModeComboBox->addItem("TRIANGLE_STRIP");
    geometryOutputModeComboBox->setCurrentIndex(0);
    connect(geometryOutputModeComboBox, SIGNAL(currentIndexChanged(int)), tab, SLOT(handleOutputTypeChanged(int)));

    QHBoxLayout *geomlayout = new QHBoxLayout();

    geomlayout->addStretch();
    geomlayout->addWidget(geometryNumVertSpin);
    geomlayout->addWidget(geometryInputModeComboBox);
    geomlayout->addWidget(geometryOutputModeComboBox);
    geomlayout->addWidget(geometryBtn);

    page34Layout->addWidget(geometryEdit);
    page34Layout->addLayout(geomlayout);

    page34->setLayout(page34Layout);

    shaderTab->addTab(page31, "Uniform");
    shaderTab->addTab(page32, "Vertex");
    shaderTab->addTab(page34, "Geometry");
    shaderTab->addTab(page33, "Fragment");
    shaderTab->addTab(page35, "TessControl");
    shaderTab->addTab(page36, "TessEval");

    shaderTabL->addWidget(shaderTab, 0, 0, Qt::AlignLeft);

    page3Layout->addLayout(shaderListL, 0, 0, Qt::AlignLeft);
    page3Layout->addLayout(shaderTabL, 0, 1, Qt::AlignLeft);
    page3->setLayout(page3Layout);

    //page4 ///////////////////////////////////////////////////////////////////////////////////////////

    QWidget *page4 = new QWidget();
    QGridLayout *page4Layout = new QGridLayout();

    // FIXME add validator

    edtR11 = new QLineEdit();
    edtR11->setValidator(new QDoubleValidator(this));
    edtR12 = new QLineEdit();
    edtR12->setValidator(new QDoubleValidator(this));
    edtR13 = new QLineEdit();
    edtR13->setValidator(new QDoubleValidator(this));
    edtTX = new QLineEdit();
    edtTX->setValidator(new QDoubleValidator(this));

    connect(edtR11, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtR11, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtR12, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtR12, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtR13, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtR13, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtTX, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtTX, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));

    edtR21 = new QLineEdit();
    edtR21->setValidator(new QDoubleValidator(this));
    edtR22 = new QLineEdit();
    edtR22->setValidator(new QDoubleValidator(this));
    edtR23 = new QLineEdit();
    edtR23->setValidator(new QDoubleValidator(this));
    edtTY = new QLineEdit();
    edtTY->setValidator(new QDoubleValidator(this));

    connect(edtR21, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtR21, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtR22, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtR22, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtR23, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtR23, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtTY, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtTY, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));

    edtR31 = new QLineEdit();
    edtR31->setValidator(new QDoubleValidator(this));
    edtR32 = new QLineEdit();
    edtR32->setValidator(new QDoubleValidator(this));
    edtR33 = new QLineEdit();
    edtR33->setValidator(new QDoubleValidator(this));
    edtTZ = new QLineEdit();
    edtTZ->setValidator(new QDoubleValidator(this));

    connect(edtR31, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtR31, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtR32, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtR32, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtR33, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtR33, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtTZ, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtTZ, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));

    edtP1 = new QLineEdit();
    edtP1->setValidator(new QDoubleValidator(this));
    edtP2 = new QLineEdit();
    edtP2->setValidator(new QDoubleValidator(this));
    edtP3 = new QLineEdit();
    edtP3->setValidator(new QDoubleValidator(this));
    edtW = new QLineEdit();
    edtW->setValidator(new QDoubleValidator(this));

    connect(edtP1, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtP1, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtP2, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtP2, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtP3, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtP3, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtW, SIGNAL(editingFinished()), this, SLOT(onMatrixChanged()));
    connect(edtW, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));

    edtRotX = new QLineEdit();
    edtRotX->setValidator(new QDoubleValidator(this));
    edtRotX->setText("");
    edtRotX->setEnabled(false);
    edtRotY = new QLineEdit();
    edtRotY->setValidator(new QDoubleValidator(this));
    edtRotY->setText("");
    edtRotY->setEnabled(false);
    edtRotZ = new QLineEdit();
    edtRotZ->setValidator(new QDoubleValidator(this));
    edtRotZ->setText("");
    edtRotZ->setEnabled(false);

    connect(edtRotX, SIGNAL(editingFinished()), this, SLOT(onDeltaRotChanged()));
    connect(edtRotX, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtRotY, SIGNAL(editingFinished()), this, SLOT(onDeltaRotChanged()));
    connect(edtRotY, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    connect(edtRotZ, SIGNAL(editingFinished()), this, SLOT(onDeltaRotChanged()));
    connect(edtRotZ, SIGNAL(textEdited(QString)), this, SLOT(onMatrixEdited()));
    
    // get the QLineEdit Widget, who called us
    QLineEdit *qle = (QLineEdit *)QObject::sender();

    // change the color to black
   // QPalette *palette = new QPalette();
   // palette->setColor(QPalette::Text, Qt::black);
    QLabel *l = new QLabel("r_1,1 =");
   
    page4Layout->addWidget(l, 0, 0, Qt::AlignRight);
    QPalette palette = l->palette();
        palette.setColor(l->backgroundRole(), Qt::black);
        palette.setColor(l->foregroundRole(), Qt::black);
        l->setPalette(palette);
    page4Layout->addWidget(edtR11, 0, 1, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("r_1,2 ="), 0, 2, Qt::AlignRight);
    page4Layout->addWidget(edtR12, 0, 3, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("r_1,3 ="), 0, 4, Qt::AlignRight);
    page4Layout->addWidget(edtR13, 0, 5, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("p_1 ="), 0, 6, Qt::AlignRight);
    page4Layout->addWidget(edtTX, 0, 7, Qt::AlignLeft);

    page4Layout->addWidget(new QLabel("r_2,1 ="), 1, 0, Qt::AlignRight);
    page4Layout->addWidget(edtR21, 1, 1, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("r_2,2 ="), 1, 2, Qt::AlignRight);
    page4Layout->addWidget(edtR22, 1, 3, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("r_2,3 ="), 1, 4, Qt::AlignRight);
    page4Layout->addWidget(edtR23, 1, 5, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("p_2 ="), 1, 6, Qt::AlignRight);
    page4Layout->addWidget(edtTY, 1, 7, Qt::AlignLeft);

    page4Layout->addWidget(new QLabel("r_3,1 ="), 2, 0, Qt::AlignRight);
    page4Layout->addWidget(edtR31, 2, 1, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("r_3,2 ="), 2, 2, Qt::AlignRight);
    page4Layout->addWidget(edtR32, 2, 3, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("r_3,3 ="), 2, 4, Qt::AlignRight);
    page4Layout->addWidget(edtR33, 2, 5, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("p_3 ="), 2, 6, Qt::AlignRight);
    page4Layout->addWidget(edtTZ, 2, 7, Qt::AlignLeft);

    page4Layout->addWidget(new QLabel("t_x ="), 3, 0, Qt::AlignRight);
    page4Layout->addWidget(edtP1, 3, 1, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("t_y ="), 3, 2, Qt::AlignRight);
    page4Layout->addWidget(edtP2, 3, 3, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("t_z ="), 3, 4, Qt::AlignRight);
    page4Layout->addWidget(edtP3, 3, 5, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("w ="), 3, 6, Qt::AlignRight);
    page4Layout->addWidget(edtW, 3, 7, Qt::AlignLeft);

    QFrame *line = new QFrame();
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    page4Layout->addWidget(line, 4, 0, 1, 8);

    page4Layout->addWidget(new QLabel("rot_x ="), 5, 0, Qt::AlignRight);
    page4Layout->addWidget(edtRotX, 5, 1, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("rot_y ="), 5, 2, Qt::AlignRight);
    page4Layout->addWidget(edtRotY, 5, 3, Qt::AlignLeft);
    page4Layout->addWidget(new QLabel("rot_z ="), 5, 4, Qt::AlignRight);
    page4Layout->addWidget(edtRotZ, 5, 5, Qt::AlignLeft);

    edtR11->setEnabled(false);
    edtR12->setEnabled(false);
    edtR13->setEnabled(false);
    edtTX->setEnabled(false);
    edtR21->setEnabled(false);
    edtR22->setEnabled(false);
    edtR23->setEnabled(false);
    edtTY->setEnabled(false);
    edtR31->setEnabled(false);
    edtR32->setEnabled(false);
    edtR33->setEnabled(false);
    edtTZ->setEnabled(false);
    edtP1->setEnabled(false);
    edtP2->setEnabled(false);
    edtP3->setEnabled(false);
    edtW->setEnabled(false);
    
    sliderH = new QSlider(page4);
    sliderP = new QSlider(page4);
    sliderR = new QSlider(page4);
    
    sliderH->setOrientation(Qt::Horizontal);
    sliderP->setOrientation(Qt::Horizontal);
    sliderR->setOrientation(Qt::Horizontal);
    
    page4Layout->addWidget(sliderH, 6, 1, Qt::AlignLeft);
    page4Layout->addWidget(sliderP, 6, 3, Qt::AlignLeft);
    page4Layout->addWidget(sliderR, 6, 5, Qt::AlignLeft);
    
    stringH = new QLineEdit(page4);
    stringP = new QLineEdit(page4);
    stringR = new QLineEdit(page4);
    page4Layout->addWidget(stringH, 7, 1, Qt::AlignLeft);
    page4Layout->addWidget(stringP, 7, 3, Qt::AlignLeft);
    page4Layout->addWidget(stringR, 7, 5, Qt::AlignLeft);

    sliderH->setMinimum(0);
    sliderH->setMaximum(1000);
    sliderP->setMinimum(0);
    sliderP->setMaximum(1000);
    sliderR->setMinimum(0);
    sliderR->setMaximum(1000);
    
    connect(sliderH, SIGNAL(valueChanged(int)), this, SLOT(sliderChanged(int)));
    connect(sliderP, SIGNAL(valueChanged(int)), this, SLOT(sliderChanged(int)));
    connect(sliderR, SIGNAL(valueChanged(int)), this, SLOT(sliderChanged(int)));
    
    
    connect(stringH, SIGNAL(editingFinished()), this, SLOT(onHPRChanged()));
    connect(stringH, SIGNAL(textEdited(QString)), this, SLOT(onHPREdited()));
    connect(stringP, SIGNAL(editingFinished()), this, SLOT(onHPRChanged()));
    connect(stringP, SIGNAL(textEdited(QString)), this, SLOT(onHPREdited()));
    connect(stringR, SIGNAL(editingFinished()), this, SLOT(onHPRChanged()));
    connect(stringR, SIGNAL(textEdited(QString)), this, SLOT(onHPREdited()));
    //connect(sliderH, SIGNAL(sliderPressed()), this, SLOT(pressed()));
    //connect(sliderH, SIGNAL(sliderReleased()), this, SLOT(released()));

    // geometryEdit = new QTextEdit();
    // geometryEdit->setWordWrapMode(QTextOption::NoWrap);

    QVBoxLayout *page4vlayout = new QVBoxLayout();

    page4vlayout->addLayout(page4Layout);
    page4->setLayout(page4vlayout);

    // put it all together now

    _tab->addTab(page1, "Material");
    _tab->addTab(page2, "Texture");
    _tab->addTab(page3, "Shader");
    _tab->addTab(page4, "Transform");

    discardButton = new QPushButton();
    discardButton->setText("Discard");
    closeButton = new QPushButton();
    closeButton->setText("Close");
    applyButton = new QPushButton();
    applyButton->setText("Apply");

    QHBoxLayout *btnlayout = new QHBoxLayout();
    btnlayout->addWidget(applyButton);
    btnlayout->addWidget(discardButton);
    btnlayout->addWidget(closeButton);

    mainLayout->addLayout(layout, 0, 0);
    mainLayout->addLayout(tabLayout, 1, 0, Qt::AlignHCenter);
    mainLayout->setRowStretch(2, 50);
    mainLayout->addLayout(btnlayout, 3, 0, Qt::AlignRight);

    connect(this, SIGNAL(updateUItem(QListWidgetItem *)), this, SLOT(handleUniformList(QListWidgetItem *)));
    connect(this, SIGNAL(getShader()), tab, SLOT(handleGetShader()));
    connect(this, SIGNAL(setUniform()), tab, SLOT(handleSetUniform()));
    connect(this, SIGNAL(apply()), tab, SLOT(handleApply()));
    connect(this, SIGNAL(closeDialog()), tab, SLOT(handleCloseDialog()));
    connect(discardButton, SIGNAL(pressed()), this, SLOT(onDiscardPressed()));
    connect(closeButton, SIGNAL(pressed()), this, SLOT(onClosePressed()));
    connect(applyButton, SIGNAL(pressed()), this, SLOT(onApplyPressed()));
    connect(childrenEdit, SIGNAL(returnPressed()), this, SLOT(onApplyPressed()));

    setLayout(mainLayout);

    viewIndex = -1;
    _shaderName = "";
    _uniformName = "";
    firstTime = true;
    emptyTextureFile = true;
    //shaderTab->hide();

    for (int i = 0; i < 16; ++i)
    {
        _matrix.data()[i] = 0;
    }

    connect(modeCB, SIGNAL(stateChanged(int)), tab, SLOT(sendChangeTextureRequest()));
    connect(transCB, SIGNAL(stateChanged(int)), tab, SLOT(sendChangeTextureRequest()));
    connect(removeMat, SIGNAL(clicked()), this, SLOT(onremoveMatClicked()));
}

void PropertyDialog::onremoveMatClicked()
{
    doRemove = true;
    emit apply();
}

void PropertyDialog::updateIcons()
{
    QColor color;

    color.setRedF(_diffuse[0]);
    color.setGreenF(_diffuse[1]);
    color.setBlueF(_diffuse[2]);
    color.setAlphaF(_diffuse[3]);

    if (color.isValid())
    {
        diffuseP->fill(color);
        diffuseL->setPixmap(*diffuseP);
    }

    color.setRedF(_specular[0]);
    color.setGreenF(_specular[1]);
    color.setBlueF(_specular[2]);
    color.setAlphaF(_specular[3]);
    if (color.isValid())
    {
        specularP->fill(color);
        specularL->setPixmap(*specularP);
    }

    color.setRedF(_ambient[0]);
    color.setGreenF(_ambient[1]);
    color.setBlueF(_ambient[2]);
    color.setAlphaF(_ambient[3]);
    if (color.isValid())
    {
        ambientP->fill(color);
        ambientL->setPixmap(*ambientP);
    }

    color.setRedF(_emissive[0]);
    color.setGreenF(_emissive[1]);
    color.setBlueF(_emissive[2]);
    color.setAlphaF(_emissive[3]);
    if (color.isValid())
    {
        emissiveP->fill(color);
        emissiveL->setPixmap(*emissiveP);
    }
}

PropertyDialog::~PropertyDialog()
{
}

void PropertyDialog::updateUniform(QString shader, QString uniform, QString value, QString textureFile)
{
    if ((shader == getShaderName()) && (uniform == getUniformName()))
    {
        if (uniformListWidget->currentItem())
        {
            UniformItem *UItem = static_cast<UniformItem *>(uniformListWidget->currentItem());
            UItem->setValue(value);
            UItem->setTextureFile(textureFile);

            emit updateUItem(uniformListWidget->currentItem());
        }
    }
}

void PropertyDialog::updateVertex(QString shader, QString value)
{
    if (shader == getShaderName())
    {
        vertexEdit->clear();
        vertexEdit->setText(value);
    }
}

void PropertyDialog::updateTessControl(QString shader, QString value)
{
    if (shader == getShaderName())
    {
        tessControlEdit->clear();
        tessControlEdit->setText(value);
    }
}

void PropertyDialog::updateTessEval(QString shader, QString value)
{
    if (shader == getShaderName())
    {
        tessEvalEdit->clear();
        tessEvalEdit->setText(value);
    }
}

void PropertyDialog::updateFragment(QString shader, QString value)
{
    if (shader == getShaderName())
    {
        fragmentEdit->clear();
        fragmentEdit->setText(value);
    }
}

void PropertyDialog::updateGeometry(QString shader, QString value)
{
    if (shader == getShaderName())
    {
        geometryEdit->clear();
        geometryEdit->setText(value);
    }
}

void PropertyDialog::setNumVert(QString shader, int value)
{
    if (shader == getShaderName())
    {
        geometryNumVertSpin->setValue(value);
    }
}

void PropertyDialog::setInputType(QString shader, int value)
{
    if (shader == getShaderName())
    {
        if (value == GL_POINTS)
            geometryInputModeComboBox->setCurrentIndex(0);
        if (value == GL_LINES)
            geometryInputModeComboBox->setCurrentIndex(1);
        if (value == MY_GL_LINES_ADJACENCY_EXT)
            geometryInputModeComboBox->setCurrentIndex(2);
        if (value == MY_GL_TRIANGLES_ADJACENCY_EXT)
            geometryInputModeComboBox->setCurrentIndex(3);
        if (value == GL_TRIANGLES)
            geometryInputModeComboBox->setCurrentIndex(4);
    }
}
void PropertyDialog::setOutputType(QString shader, int value)
{
    if (shader == getShaderName())
    {
        if (value == GL_POINTS)
            geometryOutputModeComboBox->setCurrentIndex(0);
        if (value == GL_LINE_STRIP)
            geometryOutputModeComboBox->setCurrentIndex(1);
        if (value == GL_TRIANGLE_STRIP)
            geometryOutputModeComboBox->setCurrentIndex(2);
    }
}

void PropertyDialog::onTabPressed(int index)
{
    if (firstTime)
    {
        if (index == 2)
        {
            emit getShader();

            firstTime = false;
        }
    }
}

//QString PropertyDialog::getUniformName()
//{
//   if(uniformListWidget->currentItem())
//      return uniformListWidget->currentItem()->text();
//   else
//      return "";
//}
QString PropertyDialog::getUniformValue()
{

    if (uniformListWidget->currentItem())
    {
        UniformItem *UItem = static_cast<UniformItem *>(uniformListWidget->currentItem());
        return UItem->getValue();
    }
    else
    {
        return uniformEdit->text();
    }
}

QString PropertyDialog::getUniformTextureFile()
{
    if (uniformListWidget->currentItem())
    {
        UniformItem *UItem = static_cast<UniformItem *>(uniformListWidget->currentItem());
        return UItem->getTextureFile();
    }
    else
    {
        return uniformEdit->text();
    }
}

QString PropertyDialog::getVertexValue()
{
    return vertexEdit->toPlainText();
}

QString PropertyDialog::getTessControlValue()
{
    return tessControlEdit->toPlainText();
}

QString PropertyDialog::getTessEvalValue()
{
    return tessEvalEdit->toPlainText();
}

QString PropertyDialog::getGeometryValue()
{
    return geometryEdit->toPlainText();
}
QString PropertyDialog::getFragmentValue()
{
    return fragmentEdit->toPlainText();
}

//QString PropertyDialog::getShaderName()
//{
//   if(shaderListWidget->currentItem())
//      return shaderListWidget->currentItem()->text();
//   else
//      return "";
//}
//void PropertyDialog::handleEditShader(bool on)
//{
//   if(on)
//   {
//      shaderTab->show();
//   }
//   else
//   {
//      shaderTab->hide();
//   }
//
//}
TextureItem *PropertyDialog::getTextureItem(int index)
{
    for (int i = 0; i < listWidget->count(); i++)
    {
        QListWidgetItem *item = listWidget->item(i);
        TextureItem *texItem = static_cast<TextureItem *>(item);
        if (texItem)
        {
            if (index == texItem->getIndex())
            {
                return texItem;
            }
        }
    }
    return NULL;
}

void PropertyDialog::updateView(QListWidgetItem *item)
{
    if (item)
    {
        TextureItem *texItem = static_cast<TextureItem *>(item);
        if (texItem)
        {
            viewIndex = texItem->getIndex();
            view->setPixmap(texItem->icon().pixmap(96, 96));
        }
    }
}

bool PropertyDialog::setView(int index)
{
    viewIndex = index;
    if (index >= 0)
    {
        if (getTextureItem(index))
        {
            view->setPixmap(getTextureItem(index)->icon().pixmap(96, 96));
            return true;
        }
    }
    else
    {
        view->clear();
    }
    return false;
}

void PropertyDialog::setTexMode(int index)
{
    textureModeComboBox->setCurrentIndex(index);
}

void PropertyDialog::setTexGenMode(int index)
{
    textureTexGenModeComboBox->setCurrentIndex(index);
}

int PropertyDialog::getTexNumber()
{
    return textureNumberSpin->value();
}

int PropertyDialog::getTexMode()
{
    return textureModeComboBox->currentIndex();
}

int PropertyDialog::getTexGenMode()
{
    return textureTexGenModeComboBox->currentIndex();
}

QString PropertyDialog::getFilename(int index)
{
    if (getTextureItem(index))
        return getTextureItem(index)->getPath();
    else
        return "";
}

int PropertyDialog::getCurrentIndex()
{
    return viewIndex;
}

int PropertyDialog::getNumItems()
{
    return listWidget->count();
}

void PropertyDialog::setListWidgetItem(QPixmap map, QString path, int index)
{
    new TextureItem(listWidget, map, path, index);
}

void PropertyDialog::addShader(QString name)
{
    QListWidgetItem *item = new QListWidgetItem(shaderListWidget);
    item->setText(name);
    item->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
}

void PropertyDialog::clearShaderList()
{
    shaderListWidget->clear();
}

void PropertyDialog::addUniform(QString name, QString type, QString value, QString min, QString max, QString textureFile)
{
    new UniformItem(uniformListWidget, name, type, value, min, max, textureFile);
}

void PropertyDialog::clearUniformList()
{
    uniformListWidget->clear();

    uniformEdit->clear();
    uniformColorBtn->hide();
    uniformSlider->hide();
    uniformTexFileEdit->hide();
    editLabel->hide();
    texFileEditLabel->hide();
    reloadLabel->hide();
    VecEdit1->hide();
    VecEdit2->hide();
    VecEdit3->hide();
    VecEdit4->hide();
    uniformEdit->hide();
}

void PropertyDialog::handleUniformList(QListWidgetItem *item)
{
    uniformEdit->clear();
    UniformItem *UItem = static_cast<UniformItem *>(item);
    _uniformName = UItem->text();
    uniformEdit->setText(UItem->getValue());
    uniformColorBtn->hide();
    uniformSlider->hide();
    uniformTexFileEdit->hide();
    editLabel->hide();
    texFileEditLabel->hide();
    reloadLabel->hide();
    VecEdit1->hide();
    VecEdit2->hide();
    VecEdit3->hide();
    VecEdit4->hide();
    uniformEdit->show();

    if (UItem->getType() == "vec4")
    {
        QString vec = UItem->getValue();
        QStringList veclist = vec.split(" ");
        if (veclist.size() == 4)
        {
            _uniform4[0] = veclist.at(0).toFloat();
            _uniform4[1] = veclist.at(1).toFloat();
            _uniform4[2] = veclist.at(2).toFloat();
            _uniform4[3] = veclist.at(3).toFloat();

            VecEdit1->setText(veclist.at(0));
            VecEdit2->setText(veclist.at(1));
            VecEdit3->setText(veclist.at(2));
            VecEdit4->setText(veclist.at(3));
            VecEdit1->show();
            VecEdit2->show();
            VecEdit3->show();
            VecEdit4->show();

            QColor color;
            color.setRgbF(_uniform4[0], _uniform4[1], _uniform4[2], _uniform4[3]);
            uniformColorBtn->setPalette(QPalette(color));

            uniformColorBtn->show();
            uniformEdit->hide();
        }
    }
    if (UItem->getType() == "vec3")
    {
        QString vec = UItem->getValue();
        QStringList veclist = vec.split(" ");
        if (veclist.size() == 3)
        {
            VecEdit1->setText(veclist.at(0));
            VecEdit2->setText(veclist.at(1));
            VecEdit3->setText(veclist.at(2));
            VecEdit1->show();
            VecEdit2->show();
            VecEdit3->show();
            uniformEdit->hide();
        }
    }
    if (UItem->getType() == "float")
    {
        QString min = UItem->getMin();
        QString max = UItem->getMax();
        QString value = UItem->getValue();
        int uMin;
        int uMax;
        int uValue;
        if (min.isEmpty())
            uMin = 0;
        else
            uMin = (int)(min.toFloat() * 100);
        if (max.isEmpty())
            uMax = 100;
        else
            uMax = (int)(max.toFloat() * 100);
        if (value.isEmpty())
            uValue = 0;
        else
            uValue = (int)(value.toFloat() * 100);

        uniformSlider->setMinimum(uMin);
        uniformSlider->setMaximum(uMax);
        uniformSlider->setValue(uValue);
        uniformSlider->show();
    }
    if (UItem->getType() == "int")
    {
        QString min = UItem->getMin();
        QString max = UItem->getMax();
        QString value = UItem->getValue();
        int uMin;
        int uMax;
        int uValue;
        if (min.isEmpty())
            uMin = 0;
        else
            uMin = min.toInt();
        if (max.isEmpty())
            uMax = 10;
        else
            uMax = max.toInt();
        if (value.isEmpty())
            uValue = 0;
        else
            uValue = value.toInt();
        uniformSlider->setMinimum(uMin);
        uniformSlider->setMaximum(uMax);
        uniformSlider->setValue(uValue);
        uniformSlider->show();
    }
    if (UItem->getType().contains("sampler") && (UItem->getType().endsWith("D") || UItem->getType().endsWith("Rect")))
    {
        if (firstTime && !UItem->getTextureFile().isEmpty())
            emptyTextureFile = false;

        uniformTexFileEdit->setText(UItem->getTextureFile());
        uniformTexFileEdit->show();
        editLabel->show();
        texFileEditLabel->show();
    }
}

void PropertyDialog::addSource(QString vertex, QString fragment, QString geometry, QString tessControl, QString tessEval)
{
    vertexEdit->setText(vertex);
    tessControlEdit->setText(tessControl);
    tessEvalEdit->setText(tessEval);
    fragmentEdit->setText(fragment);
    geometryEdit->setText(geometry);
}

void PropertyDialog::clearSource()
{
    vertexEdit->clear();
    tessControlEdit->clear();
    tessEvalEdit->clear();
    fragmentEdit->clear();
    geometryEdit->clear();
}

void PropertyDialog::setTextureUpdateBtn(bool state)
{
    texUpdateBtn->setEnabled(state);
}

void PropertyDialog::clearView()
{
    view->clear();
    viewIndex = -1;
}

void PropertyDialog::clearList()
{
    listWidget->clear();
}

void PropertyDialog::onDiscardPressed()
{
    updateGUI(myProps);
}

void PropertyDialog::onClosePressed()
{
    emit closeDialog();
    this->hide();
}

void PropertyDialog::onApplyPressed()
{
    emit apply();
}

void PropertyDialog::onMatrixChanged()
{
    _matrix.data()[0] = edtR11->text().toFloat();
    _matrix.data()[1] = edtR12->text().toFloat();
    _matrix.data()[2] = edtR13->text().toFloat();
    _matrix.data()[3] = edtTX->text().toFloat();
    _matrix.data()[4] = edtR21->text().toFloat();
    _matrix.data()[5] = edtR22->text().toFloat();
    _matrix.data()[6] = edtR23->text().toFloat();
    _matrix.data()[7] = edtTY->text().toFloat();
    _matrix.data()[8] = edtR31->text().toFloat();
    _matrix.data()[9] = edtR32->text().toFloat();
    _matrix.data()[10] = edtR33->text().toFloat();
    _matrix.data()[11] = edtTZ->text().toFloat();
    _matrix.data()[12] = edtP1->text().toFloat();
    _matrix.data()[13] = edtP2->text().toFloat();
    _matrix.data()[14] = edtP3->text().toFloat();
    _matrix.data()[15] = edtW->text().toFloat();

    // get the QLineEdit Widget, who called us
    QLineEdit *qle = (QLineEdit *)QObject::sender();

    // change the color to black
    QPalette *palette = new QPalette();
    palette->setColor(QPalette::Text, Qt::black);
    qle->setPalette(*palette);
    
        emit apply();
}

void PropertyDialog::onHPRChanged()
{
    float H,P,R;
    H = stringH->text().toFloat();
    P = stringP->text().toFloat();
    R = stringR->text().toFloat();
    
    sliderH->blockSignals(true);
    sliderH->setValue(H / 360.0* 1000.0 );
    sliderH->blockSignals(false);
    sliderP->blockSignals(true);
    sliderP->setValue(P / 360.0* 1000.0 );
    sliderP->blockSignals(false);
    sliderR->blockSignals(true);
    sliderR->setValue(R / 360.0* 1000.0 );
    sliderR->blockSignals(false);
    
    QMatrix4x4 rotMat,mT;
    QQuaternion quat = fromEulerAngles(P,H,R);
    rotMat.rotate(quat);
    
    rotMat = vrmlToOsg * rotMat * osgToVrml;

    float tX,tY,tZ;
    tX = _matrix.data()[12];
    tY = _matrix.data()[13];
    tZ = _matrix.data()[14];
    mT.translate(tX,tY,tZ);
    _matrix = rotMat*mT;
    
    _matrix.data()[12] = tX;
    _matrix.data()[13] = tY;
    _matrix.data()[14] = tZ;

    writeMatrixToLineEdit();
    emit apply();
}
void PropertyDialog::onHPREdited()
{
}
void PropertyDialog::sliderChanged(int ival)
{
    float H,P,R;
    H = sliderH->value() / 1000.0 * 360.0;
    P = sliderP->value() / 1000.0 * 360.0;
    R = sliderR->value() / 1000.0 * 360.0;
    char buf[1000];
    sprintf(buf,"%f",H);
    stringH->setText(buf);
    sprintf(buf,"%f",P);
    stringP->setText(buf);
    sprintf(buf,"%f",R);
    stringR->setText(buf);
    QMatrix4x4 mH,mP,mR,mT,rotMat;
    mH.rotate(H,0,0,1);
    mP.rotate(P,1,0,0);
    mR.rotate(R,0,1,0);

    QQuaternion quat = fromEulerAngles(P,H,R);
    rotMat.rotate(quat);
    
    rotMat = vrmlToOsg * rotMat * osgToVrml;

    float tX,tY,tZ;
    tX = _matrix.data()[12];
    tY = _matrix.data()[13];
    tZ = _matrix.data()[14];
    mT.translate(tX,tY,tZ);
    _matrix = rotMat*mT;
    
    _matrix.data()[12] = tX;
    _matrix.data()[13] = tY;
    _matrix.data()[14] = tZ;

    writeMatrixToLineEdit();
    emit apply();
    //TUIMainWindow::getInstance()->getStatusBar()->message(QString("Floatslider: %1").arg(value));
}
void PropertyDialog::onDeltaRotChanged()
{
    float sa = sin(edtRotZ->text().toFloat());
    float ca = cos(edtRotZ->text().toFloat());
    float sb = sin(edtRotY->text().toFloat());
    float cb = cos(edtRotY->text().toFloat());
    float sg = sin(edtRotX->text().toFloat());
    float cg = cos(edtRotX->text().toFloat());

    float m0 = _matrix.data()[0];
    float m1 = _matrix.data()[1];
    float m2 = _matrix.data()[2];

    float m4 = _matrix.data()[4];
    float m5 = _matrix.data()[5];
    float m6 = _matrix.data()[6];

    float m8 = _matrix.data()[8];
    float m9 = _matrix.data()[9];
    float m10 = _matrix.data()[10];

    _matrix.data()[0] = m8 * sb - cb * m4 * sa + ca * cb * m0;
    _matrix.data()[4] = m4 * (ca * cg - sa * sb * sg) + m0 * (ca * sb * sg + cg * sa) - cb * m8 * sg;
    _matrix.data()[8] = m0 * (sa * sg - ca * cg * sb) + m4 * (ca * sg + cg * sa * sb) + cb * cg * m8;

    _matrix.data()[1] = m9 * sb - cb * m5 * sa + ca * cb * m1;
    _matrix.data()[5] = m5 * (ca * cg - sa * sb * sg) + m1 * (ca * sb * sg + cg * sa) - cb * m9 * sg;
    _matrix.data()[9] = m1 * (sa * sg - ca * cg * sb) + m5 * (ca * sg + cg * sa * sb) + cb * cg * m9;

    _matrix.data()[2] = m10 * sb - cb * m6 * sa + ca * cb * m2;
    _matrix.data()[6] = m6 * (ca * cg - sa * sb * sg) + m2 * (ca * sb * sg + cg * sa) - cb * m10 * sg;
    _matrix.data()[10] = m2 * (sa * sg - ca * cg * sb) + m6 * (ca * sg + cg * sa * sb) + cb * cg * m10;

    writeMatrixToLineEdit();
    updateTransformSliders();

    // reset of lineedit fields
    edtRotX->setText("0");
    edtRotY->setText("0");
    edtRotZ->setText("0");

    // get the QLineEdit Widget, who called us
    QLineEdit *qle = (QLineEdit *)QObject::sender();

    // change the color to black
    QPalette *palette = new QPalette();
    palette->setColor(QPalette::Text, Qt::black);
    qle->setPalette(*palette);
    
        emit apply();
}

void PropertyDialog::onMatrixEdited()
{
    // get the QLineEdit Widget, who called us
    QLineEdit *qle = (QLineEdit *)QObject::sender();

    // change the color to red
    QPalette *palette = new QPalette();
    palette->setColor(QPalette::Text, Qt::red);
    qle->setPalette(*palette);
}

void PropertyDialog::handleUniformInt(int value)
{
    if (uniformListWidget->currentItem())
    {
        QString uValue;
        UniformItem *UItem = static_cast<UniformItem *>(uniformListWidget->currentItem());
        if (UItem->getType() == "float")
        {
            float newValue = (float)value / 100.0;
            uValue.setNum(newValue);
        }
        if (UItem->getType() == "int")
        {
            uValue.setNum(value);
        }
        UItem->setValue(uValue);
        uniformEdit->setText(uValue);
        emit setUniform();
    }
}

void PropertyDialog::handleUniformText()
{
    QString text = uniformTexFileEdit->text();
    if (uniformListWidget->currentItem())
    {
        UniformItem *UItem = static_cast<UniformItem *>(uniformListWidget->currentItem());
        UItem->setTextureFile(text);
        if (emptyTextureFile && !UItem->getTextureFile().isEmpty())
        {
            emptyTextureFile = false;
            reloadLabel->show();
        }
        emit setUniform();
    }
}

void PropertyDialog::handleSetUniformL()
{
    QString value = uniformEdit->text();
    if (uniformListWidget->currentItem())
    {
        UniformItem *UItem = static_cast<UniformItem *>(uniformListWidget->currentItem());
        QString type = UItem->getType();

        if (type == "vec4")
        {
            QStringList list = value.split(" ");
            if (list.size() == 4)
            {
                VecEdit1->setText(list.at(0));
                VecEdit2->setText(list.at(1));
                VecEdit3->setText(list.at(2));
                VecEdit4->setText(list.at(3));

                _uniform4[0] = list.at(0).toFloat();
                _uniform4[1] = list.at(1).toFloat();
                _uniform4[2] = list.at(2).toFloat();
                _uniform4[3] = list.at(3).toFloat();

                QColor color;
                color.setRgbF(_uniform4[0], _uniform4[1], _uniform4[2], _uniform4[3]);
                uniformColorBtn->setPalette(QPalette(color));
            }
        }
        if (type == "vec3")
        {
            QStringList list = value.split(" ");
            if (list.size() == 3)
            {
                VecEdit1->setText(list.at(0));
                VecEdit2->setText(list.at(1));
                VecEdit3->setText(list.at(2));
            }
        }
        if (type == "float")
        {
            int newValue = (int)(value.toDouble() * 100);
            uniformSlider->setValue(newValue);
        }
        if (type == "int")
        {
            uniformSlider->setValue(value.toInt());
        }

        UItem->setValue(value);
        if (!UItem->getTextureFile().isEmpty())
            reloadLabel->show();
        emit setUniform();
    }
}

void PropertyDialog::handleUniformVec()
{
    if (uniformListWidget->currentItem())
    {
        QString value;
        UniformItem *UItem = static_cast<UniformItem *>(uniformListWidget->currentItem());
        QString type = UItem->getType();
        if (type == "vec4")
        {
            value = VecEdit1->text() + " " + VecEdit2->text() + " " + VecEdit3->text() + " " + VecEdit4->text();

            _uniform4[0] = VecEdit1->text().toFloat();
            _uniform4[1] = VecEdit2->text().toFloat();
            _uniform4[2] = VecEdit3->text().toFloat();
            _uniform4[3] = VecEdit4->text().toFloat();

            QColor color;
            color.setRgbF(_uniform4[0], _uniform4[1], _uniform4[2], _uniform4[3]);
            uniformColorBtn->setPalette(QPalette(color));
        }
        if (type == "vec3")
        {
            value = VecEdit1->text() + " " + VecEdit2->text() + " " + VecEdit3->text();
        }
        UItem->setValue(value);
        uniformEdit->setText(value);
        emit setUniform();
    }
}

void PropertyDialog::handleUniformVec4()
{
    //  QColor initC = uniformColorBtn->palette().color(QPalette::Button);
    /*
   int r = qRed(my);
   int g = qGreen(my);
   int b = qBlue(my);
   int alpha = qAlpha ( my );
   QColor color = QColor(r,g,b,alpha);
   if(color.isValid())
   {
      _uniform4[0] = color.redF();
      _uniform4[1] = color.greenF();
      _uniform4[2] = color.blueF();
      _uniform4[3] = color.alphaF();

      uniformColorBtn->setPalette(QPalette(color));

      if(uniformListWidget->currentItem())
      {
         QString ur,ug,ub,ua;
         QString value = ur.setNum(_uniform4[0]) + " " + ug.setNum(_uniform4[1]) + " " + ub.setNum(_uniform4[2]) + " " + ua.setNum(_uniform4[3]);
         VecEdit1->setText(ur.setNum(_uniform4[0]));
         VecEdit2->setText(ug.setNum(_uniform4[1]));
         VecEdit3->setText(ub.setNum(_uniform4[2]));
         VecEdit4->setText(ua.setNum(_uniform4[3]));

         UniformItem * UItem = static_cast<UniformItem *>(uniformListWidget->currentItem());
         UItem->setValue(value);
         uniformEdit->setText(value);
         emit setUniform();
      }
    	emit apply();
   }
*/
}

void PropertyDialog::onColorChanged(const QColor &color)
{
    if (color.isValid())
    {
        if (diffuseCB->isChecked())
        {
            _diffuse[0] = color.redF();
            _diffuse[1] = color.greenF();
            _diffuse[2] = color.blueF();
            _diffuse[3] = color.alphaF();
            //diffuseP->fill(color);
        }
        if (specularCB->isChecked())
        {
            _specular[0] = color.redF();
            _specular[1] = color.greenF();
            _specular[2] = color.blueF();
            _specular[3] = color.alphaF();
        }
        if (ambientCB->isChecked())
        {
            _ambient[0] = color.redF();
            _ambient[1] = color.greenF();
            _ambient[2] = color.blueF();
            _ambient[3] = color.alphaF();
        }
        if (emissiveCB->isChecked())
        {
            _emissive[0] = color.redF();
            _emissive[1] = color.greenF();
            _emissive[2] = color.blueF();
            _emissive[3] = color.alphaF();
        }
        updateIcons();
        emit setUniform();
        emit apply();
    }
}


void PropertyDialog::writeMatrixToLineEdit()
{
    edtR11->setText(QString::number(_matrix.data()[0]));
    edtR11->setEnabled(true);
    edtR12->setText(QString::number(_matrix.data()[1]));
    edtR12->setEnabled(true);
    edtR13->setText(QString::number(_matrix.data()[2]));
    edtR13->setEnabled(true);
    edtTX->setText(QString::number(_matrix.data()[3]));
    edtTX->setEnabled(true);

    edtR21->setText(QString::number(_matrix.data()[4]));
    edtR21->setEnabled(true);
    edtR22->setText(QString::number(_matrix.data()[5]));
    edtR22->setEnabled(true);
    edtR23->setText(QString::number(_matrix.data()[6]));
    edtR23->setEnabled(true);
    edtTY->setText(QString::number(_matrix.data()[7]));
    edtTY->setEnabled(true);

    edtR31->setText(QString::number(_matrix.data()[8]));
    edtR31->setEnabled(true);
    edtR32->setText(QString::number(_matrix.data()[9]));
    edtR32->setEnabled(true);
    edtR33->setText(QString::number(_matrix.data()[10]));
    edtR33->setEnabled(true);
    edtTZ->setText(QString::number(_matrix.data()[11]));
    edtTZ->setEnabled(true);

    edtP1->setText(QString::number(_matrix.data()[12]));
    edtP1->setEnabled(true);
    edtP2->setText(QString::number(_matrix.data()[13]));
    edtP2->setEnabled(true);
    edtP3->setText(QString::number(_matrix.data()[14]));
    edtP3->setEnabled(true);
    edtW->setText(QString::number(_matrix.data()[15]));
    edtW->setEnabled(true);

    edtRotX->setText("0");
    edtRotY->setText("0");
    edtRotZ->setText("0");
    edtRotX->setEnabled(true);
    edtRotY->setEnabled(true);
    edtRotZ->setEnabled(true);

}

void PropertyDialog::updateTransformSliders()
{
 
    QMatrix3x3 rotMat;

    QMatrix4x4 mat = _matrix;
    mat.data()[12] = 0;
    mat.data()[13] = 0;
    mat.data()[14] = 0;

    mat = osgToVrml * mat * vrmlToOsg;
    for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
        rotMat(i,j) = mat(i,j);
    QQuaternion quat = fromRotationMatrix(rotMat);
    float h,p,r;
    getEulerAngles(quat,&p,&h,&r);
    
    char buf[1000];
    sprintf(buf,"%f",h);
    stringH->setText(buf);
    sprintf(buf,"%f",p);
    stringP->setText(buf);
    sprintf(buf,"%f",r);
    stringR->setText(buf);
    if(h<0)
        h = 360 + h;
    if(p<0)
        p = 360 + p;
    if(r<0)
        r = 360 + r;
    sliderH->blockSignals(true);
    sliderH->setValue(h*1000/360);
    sliderH->blockSignals(false);
    sliderP->blockSignals(true);
    sliderP->setValue(p*1000/360);
    sliderP->blockSignals(false);
    sliderR->blockSignals(true);
    sliderR->setValue(r*1000/360);
    sliderR->blockSignals(false);

}

void PropertyDialog::updateGUI(itemProps prop)
{
    _name->setText(prop.name);
    _type->setText(prop.type);

    childrenEdit->setText("");
    _num->setText(prop.numChildren);

    if (prop.mode)
    {
        modeCB->setCheckState(Qt::Checked);
    }
    else
    {
        modeCB->setCheckState(Qt::Unchecked);
    }
    if (prop.trans)
    {
        transCB->setCheckState(Qt::Checked);
    }
    else
    {
        transCB->setCheckState(Qt::Unchecked);
    }

    // removeMat->setCheckState(Qt::Unchecked);

    R_line->setChecked(true);

    if (prop.type == "Switch")
    {
        numL->show();
        _num->show();

        R_line->setToolTip(QString(" from 1 to %1 \n e.g.: 1;3;5-12 \n 0 to hide all").arg(prop.numChildren));
        R_line->show();
        childrenEdit->setToolTip(QString(" from 1 to %1 \n e.g.: 1;3;5-12 \n 0 to hide all").arg(prop.numChildren));
        childrenEdit->show();

        R_all->show();
    }
    else
    {
        numL->hide();
        _num->hide();

        R_line->hide();
        childrenEdit->hide();

        R_all->hide();
    }

    if (prop.type == "MatrixTransform")
    {
        for (int i = 0; i < 16; ++i)
        {
            _matrix.data()[i] = prop.matrix[i];
        }
        writeMatrixToLineEdit();
        updateTransformSliders();
    }
    else
    {
        edtR11->clear();
        edtR11->setEnabled(false);
        edtR12->clear();
        edtR12->setEnabled(false);
        edtR13->clear();
        edtR13->setEnabled(false);
        edtTX->clear();
        edtTX->setEnabled(false);

        edtR21->clear();
        edtR21->setEnabled(false);
        edtR22->clear();
        edtR22->setEnabled(false);
        edtR23->clear();
        edtR23->setEnabled(false);
        edtTY->clear();
        edtTY->setEnabled(false);

        edtR31->clear();
        edtR31->setEnabled(false);
        edtR32->clear();
        edtR32->setEnabled(false);
        edtR33->clear();
        edtR33->setEnabled(false);
        edtTZ->clear();
        edtTZ->setEnabled(false);

        edtP1->clear();
        edtP1->setEnabled(false);
        edtP2->clear();
        edtP2->setEnabled(false);
        edtP3->clear();
        edtP3->setEnabled(false);
        edtW->clear();
        edtW->setEnabled(false);

        edtRotX->setText("");
        edtRotY->setText("");
        edtRotZ->setText("");
        edtRotX->setEnabled(false);
        edtRotY->setEnabled(false);
        edtRotZ->setEnabled(false);
    }

    /*if(prop.type == "Geode")
   {
      texApplyButton->setEnabled(true);
      texRemoveBtn->setEnabled(true);
   }
   else
   {
      texApplyButton->setEnabled(false);
      texRemoveBtn->setEnabled(false);
   }*/

    for (int i = 0; i < 4; i++)
    {
        _diffuse[i] = prop.diffuse[i];
        _specular[i] = prop.specular[i];
        _ambient[i] = prop.ambient[i];
        _emissive[i] = prop.emissive[i];
    }

    QColor color;

    if (diffuseCB->isChecked())
    {
        color.setRedF(_diffuse[0]);
        color.setGreenF(_diffuse[1]);
        color.setBlueF(_diffuse[2]);
        color.setAlphaF(_diffuse[3]);
    }
    else if (specularCB->isChecked())
    {
        color.setRedF(_specular[0]);
        color.setGreenF(_specular[1]);
        color.setBlueF(_specular[2]);
        color.setAlphaF(_specular[3]);
    }
    else if (ambientCB->isChecked())
    {
        color.setRedF(_ambient[0]);
        color.setGreenF(_ambient[1]);
        color.setBlueF(_ambient[2]);
        color.setAlphaF(_ambient[3]);
    }
    else if (emissiveCB->isChecked())
    {
        color.setRedF(_emissive[0]);
        color.setGreenF(_emissive[1]);
        color.setBlueF(_emissive[2]);
        color.setAlphaF(_emissive[3]);
    }

    if (!(color.red() == 0 && color.blue() == 0 && color.green() == 0))
        colorchooser->setColor(color);
    updateIcons();
}

bool PropertyDialog::setProperties(itemProps prop)
{
    updateGUI(prop);
    myProps = prop;

    return true;
}

itemProps PropertyDialog::getProperties()
{
    itemProps prop;
    prop.name = _name->text();
    prop.type = _type->text();

    if (modeCB->checkState())
    {
        prop.mode = 1;
    }
    else
    {
        prop.mode = 0;
    }
    if (transCB->checkState())
    {
        prop.trans = 1;
    }
    else
    {
        prop.trans = 0;
    }
    if (doRemove)
    {
        prop.remove = 1;
        doRemove = false;
    }
    else
    {
        prop.remove = 0;
    }
    if (prop.type == "Switch")
    {
        if (childrenEdit->text().isEmpty())
            prop.children = "NOCHANGE";
        else
            prop.children = childrenEdit->text();

        if (R_all == R_group->checkedButton())
        {
            prop.allChildren = 1;
        }
        else
        {
            prop.allChildren = 0;
        }
    }
    else
    {
        prop.children = "";
        prop.allChildren = 0;
    }
    if (prop.type == "MatrixTransform")
    {
        for (int i = 0; i < 16; ++i)
        {
            prop.matrix[i] = _matrix.data()[i];
        }
    }
    else
    {
        for (int i = 0; i < 16; ++i)
        {
            prop.matrix[i] = 0;
        }
    }
    for (int i = 0; i < 4; i++)
    {
        prop.diffuse[i] = _diffuse[i];
        prop.specular[i] = _specular[i];
        prop.ambient[i] = _ambient[i];
        prop.emissive[i] = _emissive[i];
    }

    for (int i = 0; i < 16; ++i)
    {
        prop.matrix[i] = _matrix.data()[i];
    }

    return prop;
}

TextureItem::TextureItem(QListWidget *parent, QPixmap map, QString path, int index)
    : QListWidgetItem(parent)
{
    setIcon(QIcon(map));
    setToolTip(path);
    setTextAlignment(Qt::AlignHCenter);
    setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    _path = path;
    _index = index;
}

TextureItem::~TextureItem()
{
}

UniformItem::UniformItem(QListWidget *parent, QString name, QString type, QString value, QString min, QString max, QString textureFile)
    : QListWidgetItem(parent)
{
    setText(name);
    _type = type;
    _value = value;
    _min = min;
    _max = max;
    _textureFile = textureFile;
}

UniformItem::~UniformItem()
{
}
