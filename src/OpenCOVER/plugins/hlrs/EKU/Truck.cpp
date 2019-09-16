#include <Truck.h>

using namespace opencover;
size_t Truck:: count = 0;

Truck::Truck(osg::Vec3 pos):pos(pos)
{

    count++;
    fprintf(stderr, "new Truck\n");
    osg::Box *truck = new osg::Box(pos,length,width,height);
    truckDrawable = new osg::ShapeDrawable(truck,hint.get());

    // Declare a instance of the geode class:
    truckGeode = new osg::Geode();
    truckGeode->setName("Truck" +std::to_string(Truck::count));

    osg::Vec4 _color;
    _color.set(1.0, 0.0, 0.0, 1.0);
    truckDrawable->setColor(_color);
    truckDrawable->setUseDisplayList(false);
    osg::StateSet *mystateSet = truckGeode->getOrCreateStateSet();
    setStateSet(mystateSet);

    // Add the unit cube drawable to the geode:
    truckGeode->addDrawable(truckDrawable);


    text = new osgText::Text;
  //  text->setName("Text");
  //  text->setText("Truck"+std::to_string(Truck::count));
    //text->setColor()
  //  text->setCharacterSize(4);
  //  text->setPosition(truck->getCenter());

    truckGeode->addChild(text.get());
}

void Truck::setStateSet(osg::StateSet *stateSet)
{
    osg::Material *material = new osg::Material();
    material->setColorMode(osg::Material::DIFFUSE);
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));
  /*  osg::LightModel *defaultLm;
    defaultLm = new osg::LightModel();
    defaultLm->setLocalViewer(true);
    defaultLm->setTwoSided(true);
    defaultLm->setColorControl(osg::LightModel::SINGLE_COLOR);
    */stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
  //  stateSet->setAttributeAndModes(defaultLm, osg::StateAttribute::ON);
}
Truck::~Truck()
{
    fprintf(stderr, "Removed Truck\n");
}

bool::Truck::destroy()
{
    //is this function necessary or do it in Destructor?
    //free memory space???
    //call desctuctor???
    cover->getObjectsRoot()->removeChild(truckGeode.get());
    return true;
}

void Truck::updateColor()
{
    truckDrawable->setColor(osg::Vec4(1., 1., 0., 1.0f));
}

void Truck::resetColor()
{
    truckDrawable->setColor(osg::Vec4(1.0, 0.0, 0.0, 1.0f));
}
