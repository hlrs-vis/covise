/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
//			.h File
//
// * Description    : MoleculeViewer plugin module for the Cover Covise Renderer
//                    Reads Molecule Structures based on the Jorgensen Model
//                    The data is provided from the Itt / University Stuttgart
//
// * Class(es)      :
//
// * inherited from :
//
// * Author  : Thilo Krueger
//
// * History : started 6.7.2001
//
// **************************************************************************

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;
#include "Particles.h"
#include <cover/coTabletUI.h>
#include <QStringList>
#include <QMap>

class ParticleViewer : public coVRPlugin, public coTUIListener
{

private:
    void tabletPressEvent(coTUIElement *);
    void tabletEvent(coTUIElement *);

    std::list<Particles *> particles;

    int loadData(std::string particlepath, osg::Group *parent);
    void unloadData(std::string particlepath);
    void deleteColorMap(const QString &name);
    void readConfig();
    QStringList mapNames;
    QMap<QString, int> mapSize;
    QMap<QString, float *> mapValues;

    typedef float FlColor[5];
    int currentMap;
    int aCurrentMap;
    static ParticleViewer *plugin;

public:
    // Constructor
    ParticleViewer();

    // Destructor
    ~ParticleViewer();
    static ParticleViewer * instance()
    {
        return plugin;
    };
    bool init();

    void setTimestep(int t);
    float getMinVal();
    float getMaxVal();
    float getRadius();
    float getAMinVal();
    float getAMaxVal();
    float getARadius();
    void setParticlesColorMap(const std::string &name)
    {
        mapChoice->setSelectedText(name);
        currentMap = mapNames.indexOf(QString(name.c_str()));
    };
    void setArrowsColorMap(const std::string &name)
    {
        aMapChoice->setSelectedText(name);
        aCurrentMap = mapNames.indexOf(QString(name.c_str()));
    };
    void setParticlesValue(const std::string &name)
    {
        valChoice->setSelectedText(name);
    };
    void setArrowsValue(const std::string &name)
    {
        aValChoice->setSelectedText(name);
    };
    void setParticlesMin(float min)
    {
        mapMin->setValue(min);
    };
    void setParticlesMax(float max)
    {
        mapMax->setValue(max);
    };
    void setParticlesRadius(float radius)
    {
        radiusEdit->setValue(radius);
    };
    void setArrowsMin(float min)
    {
        aMapMin->setValue(min);
    };
    void setArrowsMax(float max)
    {
        aMapMax->setValue(max);
    };
    void setArrowsRadius(float radius)
    {
        aRadiusEdit->setValue(radius);
    };
    void setArrowsRadiusValue(const std::string &name)
    {
        aRadChoice->setSelectedText(name);
    };
    void setParticlesRadiusValue(const std::string &name)
    {
        radChoice->setSelectedText(name);
    };
    void updateColors();

    void preFrame(); // Update function , called each frame

    static int loadFile(const char *name, osg::Group *parent, const char *covise_key);
    static int unloadFile(const char *name, const char *covise_key);

    coTUITab *particleTab;
    coTUILabel *particleLabel;
    coTUIFrame *particleFrame;
    coTUILabel *arrowLabel;
    coTUIFrame *arrowFrame;
    coTUIComboBox *mapChoice;
    coTUIComboBox *valChoice;
    coTUIComboBox *radChoice;
    coTUILabel *mapLabel;
    coTUILabel *mapMinLabel;
    coTUILabel *mapMaxLabel;
    coTUILabel *radiusLabel;
    coTUIEditFloatField *mapMin;
    coTUIEditFloatField *mapMax;
    coTUIEditFloatField *radiusEdit;

    coTUIComboBox *aMapChoice;
    coTUIComboBox *aValChoice;
    coTUIComboBox *aRadChoice;
    coTUILabel *aMapLabel;
    coTUILabel *aMapMinLabel;
    coTUILabel *aMapMaxLabel;
    coTUILabel *aRadiusLabel;
    coTUIEditFloatField *aMapMin;
    coTUIEditFloatField *aMapMax;
    coTUIEditFloatField *aRadiusEdit;
    osg::Vec4 getColor(float pos, int mode);
};
