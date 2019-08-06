#include <Truck.h>

using namespace opencover;

Truck::Truck(osg::Vec3 pos):pos(pos)
{


    fprintf(stderr, "new Truck\n");
    osg::Box *truck = new osg::Box(pos,length,width,height);
    osg::ShapeDrawable *truckDrawable = new osg::ShapeDrawable(truck);

    // Declare a instance of the geode class:
    truckGeode = new osg::Geode();
    truckGeode->setName("Truck");

    osg::Vec4 _color;
    _color.set(0.0, 0.0, 1.0, 1.0);
    truckDrawable->setColor(_color);
    truckDrawable->setUseDisplayList(false);

    // Add the unit cube drawable to the geode:
    truckGeode->addDrawable(truckDrawable);

    cover->getObjectsRoot()->addChild(truckGeode.get());

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



