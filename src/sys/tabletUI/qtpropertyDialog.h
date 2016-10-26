/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
 **
 */
#ifndef QT_PROPERTY_DIALOG_H
#define QT_PROPERTY_DIALOG_H

#include <QDialog>
#include <QPushButton>
#include <QRadioButton>
#include <QButtonGroup>
#include <QLayout>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>
#include <QSpacerItem>
#include <QSpinBox>
#include <QGroupBox>
#include <QColorDialog>
#include <QColor>
#include <QTabWidget>
#include <QListWidget>
#include <QTextEdit>
#include <QSlider>
#include <QAction>
#include <QMatrix4x4>
//#include <util/coRestraint.h>
#include "color/MEColorChooser.h"
#include "TUISGBrowserTab.h"

class TextureItem;
class UniformItem;
class TUISGBrowserTab;

struct itemProps
{
    QString name;
    QString type;
    QString children;
    int mode; // rendermode
    QString numChildren;
    int allChildren;
    float diffuse[4];
    float specular[4];
    float ambient[4];
    float emissive[4];
    int remove; //remove Material
    int trans;
    float matrix[16];
};

class TextureItem : public QListWidgetItem
{
public:
    TextureItem(QListWidget *, QPixmap map, QString path, int index);
    ~TextureItem();
    QString getPath()
    {
        return _path;
    };
    int getIndex()
    {
        return _index;
    };

private:
    QString _path;
    int _index;
};
class UniformItem : public QListWidgetItem
{
public:
    UniformItem(QListWidget *, QString, QString, QString, QString, QString, QString);
    ~UniformItem();
    QString getType()
    {
        return _type;
    };
    QString getValue()
    {
        return _value;
    };
    QString getMin()
    {
        return _min;
    };
    QString getMax()
    {
        return _max;
    };
    QString getTextureFile()
    {
        return _textureFile;
    };
    void setValue(QString value)
    {
        _value = value;
    };
    void setTextureFile(QString textureFile)
    {
        _textureFile = textureFile;
    };

private:
    QString _type;
    QString _value;
    QString _min;

    QString _max;
    QString _textureFile;
};

class PropertyDialog : public QDialog
{
    Q_OBJECT

public:
    PropertyDialog(TUISGBrowserTab *tab, QWidget *parent = 0);
    ~PropertyDialog();

    bool setProperties(itemProps);
    void updateGUI(itemProps);
    itemProps getProperties();
    void setTextureUpdateBtn(bool);
    void setListWidgetItem(QPixmap, QString, int);
    int getNumItems();
    int getCurrentIndex();
    QString getFilename(int);
    void setTexMode(int);
    void setTexGenMode(int);
    int getTexNumber();
    int getTexMode();
    int getTexGenMode();
    void clearList();
    bool setView(int);
    void clearView();
    void addShader(QString);
    void clearShaderList();
    void addUniform(QString, QString, QString, QString, QString, QString);
    void clearUniformList();
    void addSource(QString, QString, QString, QString, QString);
    void clearSource();
    QString getShaderName()
    {
        return _shaderName;
    };
    void setShaderName(QString name)
    {
        _shaderName = name;
    };
    QString getUniformName()
    {
        return _uniformName;
    };
    QString getUniformValue();
    QString getUniformTextureFile();
    QString getVertexValue();
    QString getTessControlValue();
    QString getTessEvalValue();
    QString getFragmentValue();
    QString getGeometryValue();
    void updateUniform(QString, QString, QString, QString);
    void updateVertex(QString, QString);
    void updateTessControl(QString, QString);
    void updateTessEval(QString, QString);
    void updateFragment(QString, QString);
    void updateGeometry(QString, QString);
    void setNumVert(QString, int);
    void setInputType(QString, int);
    void setOutputType(QString, int);
    TextureItem *getTextureItem(int index);

private:
    MEColorChooser *colorchooser;

    float _diffuse[4];
    float _specular[4];
    float _ambient[4];
    float _emissive[4];
    float _uniform4[4];
    QMatrix4x4 _matrix;

    QString _shaderName;
    QString _uniformName;

    int viewIndex;
    bool firstTime;
    bool doRemove;
    bool emptyTextureFile;

    itemProps myProps;

    QGridLayout *layout;

    QPushButton *discardButton;
    QPushButton *closeButton;
    QPushButton *applyButton;

    QCheckBox *diffuseCB;
    QCheckBox *specularCB;
    QCheckBox *ambientCB;
    QCheckBox *emissiveCB;
    QPixmap *diffuseP;
    QPixmap *specularP;
    QPixmap *ambientP;
    QPixmap *emissiveP;
    QLabel *diffuseL; //enthalten QPixmap
    QLabel *specularL;
    QLabel *ambientL;
    QLabel *emissiveL;
    void updateIcons();
    QLineEdit *childrenEdit;

    QCheckBox *modeCB;
    QCheckBox *transCB;
    QPushButton *removeMat;

    QRadioButton *R_line;
    QRadioButton *R_all;
    QButtonGroup *R_group;

    QLabel *_name;
    QLabel *_type;
    QLabel *_num;
    QLabel *numL;
    QLabel *view;
    QListWidget *listWidget;
    QPushButton *texUpdateBtn;
    QPushButton *texApplyButton;
    QPushButton *texRemoveBtn;

    QSpinBox *textureNumberSpin;
    QComboBox *textureModeComboBox;
    QComboBox *textureTexGenModeComboBox;

    QTabWidget *shaderTab;
    QListWidget *shaderListWidget;
    QPushButton *shaderGetBtn;
    QPushButton *shaderSetBtn;
    QPushButton *shaderRemoveBtn;
    QPushButton *shaderStoreBtn;

    //QPushButton         *shaderEditBtn;
    QListWidget *uniformListWidget;
    QPushButton *uniformAddBtn;
    QLabel *editLabel;
    QLineEdit *uniformEdit;
    QLabel *texFileEditLabel;
    QLineEdit *uniformTexFileEdit;
    QPushButton *uniformColorBtn;
    QSlider *uniformSlider;
    QLabel *reloadLabel;

    QLineEdit *VecEdit1;
    QLineEdit *VecEdit2;
    QLineEdit *VecEdit3;
    QLineEdit *VecEdit4;

    QTextEdit *vertexEdit;
    QPushButton *vertexBtn;
    QTextEdit *tessControlEdit;
    QPushButton *tessControlBtn;
    QTextEdit *tessEvalEdit;
    QPushButton *tessEvalBtn;
    QTextEdit *fragmentEdit;
    QPushButton *fragmentBtn;
    QTextEdit *geometryEdit;
    QPushButton *geometryBtn;
    QComboBox *geometryInputModeComboBox;
    QComboBox *geometryOutputModeComboBox;
    QSpinBox *geometryNumVertSpin;

    QLineEdit *edtR11; // Textedit for Page4 of Tab
    QLineEdit *edtR12;
    QLineEdit *edtR13;
    QLineEdit *edtTX;
    QLineEdit *edtR21;
    QLineEdit *edtR22;
    QLineEdit *edtR23;
    QLineEdit *edtTY;
    QLineEdit *edtR31;
    QLineEdit *edtR32;
    QLineEdit *edtR33;
    QLineEdit *edtTZ;
    QLineEdit *edtP1;
    QLineEdit *edtP2;
    QLineEdit *edtP3;
    QLineEdit *edtW;
    QLineEdit *edtRotX;
    QLineEdit *edtRotY;
    QLineEdit *edtRotZ;
    QSlider *sliderH;
    QSlider *sliderP;
    QSlider *sliderR;
    QLineEdit *stringH;
    QLineEdit *stringP;
    QLineEdit *stringR;

    QWidget *page3;
    QMatrix4x4 vrmlToOsg;
    QMatrix4x4 osgToVrml;

    void writeMatrixToLineEdit();
    void updateTransformSliders();

signals:
    void setUniform();
    void apply();
    void closeDialog();
    void getShader();
    void updateUItem(QListWidgetItem *);

private slots:
    void onremoveMatClicked();
    void onColorChanged(const QColor &);
    //void     handleEditShader(bool);
    void onTabPressed(int);
    void handleUniformInt(int);
    void handleUniformText();
    void handleUniformVec();
    void handleUniformVec4();
    void handleSetUniformL();
    void handleUniformList(QListWidgetItem *);
    void updateView(QListWidgetItem *);
    void onDiscardPressed();
    void onClosePressed();
    void onApplyPressed();
    void onMatrixChanged();
    void onMatrixEdited();
    void onHPRChanged();
    void onHPREdited();
    void onDeltaRotChanged();
void sliderChanged(int ival);
};
#endif
