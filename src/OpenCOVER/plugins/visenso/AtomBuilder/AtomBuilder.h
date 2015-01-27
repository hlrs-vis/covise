/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ATOM_BUILDER_H
#define _ATOM_BUILDER_H

#define MAX_PROTONS 15
#define MAX_NEUTRONS 15
#define MAX_ELECTRONS 20

#define PROTONS_POS_X 350
#define PROTONS_POS_Y 0
#define PROTONS_POS_Z 80
#define NEUTRONS_POS_X 350
#define NEUTRONS_POS_Y 0
#define NEUTRONS_POS_Z 0
#define ELECTRONS_POS_X 350
#define ELECTRONS_POS_Y 0
#define ELECTRONS_POS_Z -80
#define ATOM_CENTER_X -500
#define ATOM_CENTER_Y 0
#define ATOM_CENTER_Z 0
#define ATOM_NUCLEUS_RADIUS 150
#define ATOM_KSHELL_RADIUS 200
#define ATOM_LSHELL_RADIUS 250
#define ATOM_MSHELL_RADIUS 300

#include "ElementDatabase.h"

#include <osg/Vec3>
#include <osg/Group>
#include <list>
#include <osgText/Text>
namespace opencover
{
class coHud;
}

using namespace opencover;
using namespace covise;
class NucleonInteractor;
class ElectronInteractor;
class ElementaryParticleInteractor;
class CheckButton;

class AtomBuilder
{
public:
    AtomBuilder();
    ~AtomBuilder();
    void setElement(Element el);
    void update();
    bool getStatus()
    {
        return forwardOk_;
    };
    void showErrorPanel();
    void show(bool value);

private:
    std::vector<ElementaryParticleInteractor *> protons_;
    std::vector<ElementaryParticleInteractor *> neutrons_;
    std::vector<ElementaryParticleInteractor *> electrons_;
    std::vector<osg::Vec3> protonPositions_;
    std::vector<osg::Vec3> neutronPositions_;

    std::list<ElementaryParticleInteractor *> protonsOutside_;

    std::list<ElementaryParticleInteractor *> electronsInKShell_;
    std::list<ElementaryParticleInteractor *> electronsInLShell_;
    std::list<ElementaryParticleInteractor *> electronsInMShell_;

    int oldElectronsInKShellSize_, oldElectronsInLShellSize_, oldElectronsInMShellSize_;
    osg::ref_ptr<osg::Group> group_;
    Element *currentElement_;
    float atomNucleusRadius_, atomKShellRadius_, atomLShellRadius_, atomMShellRadius_;
    osg::Vec3 normal_;

    bool check(std::vector<ElementaryParticleInteractor *> particles, int numParticles, float radius);
    bool check(std::vector<ElementaryParticleInteractor *> particles, int numParticles, float radius1, float radius2);

    void resetParticles();
    coHud *hud_; // hud for messages
    //int showHud_;
    float hudTime_;
    bool forwardOk_;
    float sizeNucleon_;
    float px_, py_, pz_, nx_, ny_, nz_, ex_, ey_, ez_;

    void makeText(const std::string &t, float s, osg::Vec3 p);
    void makeDescription(const std::string &heading, int np, int nn, int ne, float s, osg::Vec3 p);
    void updateDescription(const std::string &heading, int np, int nn, int ne);
    osgText::Text *descrText_;
    int nP_, nN_, nKE_, nLE_, nME_;
    CheckButton *checkButton_;
    //int showCheck_;
    float checkTime_;
};

#endif
