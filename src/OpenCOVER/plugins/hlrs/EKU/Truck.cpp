#include <Truck.h>

using namespace opencover;
size_t Truck:: count = 0;

Truck::Truck(osg::Vec3 pos):pos(pos)
{

    count++;
    fprintf(stderr, "new Truck\n");
    osg::Box *truck = new osg::Box(pos,length,width,height);
    osg::ShapeDrawable *truckDrawable = new osg::ShapeDrawable(truck);

    // Declare a instance of the geode class:
    truckGeode = new osg::Geode();
    truckGeode->setName("Truck" +std::to_string(Truck::count));

    osg::Vec4 _color;
    _color.set(0.0, 0.0, 1.0, 1.0);
    truckDrawable->setColor(_color);
    truckDrawable->setUseDisplayList(false);

    // Add the unit cube drawable to the geode:
    truckGeode->addDrawable(truckDrawable);


    text = new osgText::Text;
    text->setName("Text");
    text->setText("Truck"+std::to_string(Truck::count));
    //text->setColor()
    text->setCharacterSize(4);
    text->setPosition(truck->getCenter());

    truckGeode->addChild(text.get());
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



