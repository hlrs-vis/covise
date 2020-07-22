#include <osg/Texture2D>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgGA/EventVisitor>
#include <osgGA/StateSetManipulator>
#include <osgGA/TrackballManipulator>
#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/Viewer>

#include "IGizmo.h"

#ifndef GIZMO_DRAWABLE_H
#define GIZMO_DRAWABLE_H

/** GizmoDrawable - 
*/
class GizmoDrawable : public osg::Drawable
{
public:
    struct GizmoEventCallback : public osg::Drawable::EventCallback
    {
        virtual void event( osg::NodeVisitor* nv, osg::Drawable* drawable )
        {
            osgGA::EventVisitor* ev = static_cast<osgGA::EventVisitor*>( nv );
            GizmoDrawable* gizmoDrawable = dynamic_cast<GizmoDrawable*>( drawable );
            if ( !ev || !gizmoDrawable ) return;
            
            const osgGA::EventQueue::Events& events = ev->getEvents();
            std::cout<<"number of Events: "<< ev->getEvents().size() << std::endl;
            
            for ( osgGA::EventQueue::Events::const_iterator itr=events.begin();
                  itr!=events.end(); ++itr )
            {
                const osgGA::GUIEventAdapter* ea = (*itr)->asGUIEventAdapter();
                int x = ea->getX(), y = ea->getY();
                std::cout<<"xPos: "<< x <<" yPos: "<< y <<std::endl;
                if ( gizmoDrawable->getGizmoObject() )
                        gizmoDrawable->getGizmoObject()->OnMouseMove( x, y );   
                if ( ea->getMouseYOrientation()==osgGA::GUIEventAdapter::Y_INCREASING_UPWARDS )
                    y = ea->getWindowHeight() - y;
                switch ( ea->getEventType() )
                {
                case osgGA::GUIEventAdapter::SCROLL:
                    // you would have other methods to select among gizmos, this is only an example
                    {
                        //int mode = gizmoDrawable->getGizmoMode();
                        //if ( ea->getScrollingMotion()==osgGA::GUIEventAdapter::SCROLL_UP )
                        //    mode = (mode==GizmoDrawable::NO_GIZMO ? GizmoDrawable::SCALE_GIZMO : mode-1);
                        //else if ( ea->getScrollingMotion()==osgGA::GUIEventAdapter::SCROLL_DOWN )
                        //    mode = (mode==GizmoDrawable::SCALE_GIZMO ? GizmoDrawable::NO_GIZMO : mode+1);
                        //gizmoDrawable->setGizmoMode( (GizmoDrawable::Mode)mode );
                    }
                    break;
                case osgGA::GUIEventAdapter::PUSH:
                    if ( gizmoDrawable->getGizmoObject() )
                        {
                            gizmoDrawable->getGizmoObject()->OnMouseDown( x, y );
                            std::cout<<"Mouse Down"<<std::endl;
                        }
                    break;
                case osgGA::GUIEventAdapter::RELEASE:
                    if ( gizmoDrawable->getGizmoObject() )
                        gizmoDrawable->getGizmoObject()->OnMouseUp( x, y );
                    break;
                case osgGA::GUIEventAdapter::MOVE:
                case osgGA::GUIEventAdapter::DRAG:
                    if ( gizmoDrawable->getGizmoObject() )
                        gizmoDrawable->getGizmoObject()->OnMouseMove( x, y );
                    break;
                case osgGA::GUIEventAdapter::RESIZE:
                    gizmoDrawable->setScreenSize( ea->getWindowWidth(), ea->getWindowHeight() );
                    break;
                case osgGA::GUIEventAdapter::FRAME:
                    gizmoDrawable->applyTransform();
                    break;
                default: break;
                }
            }
        }
    };
    
    GizmoDrawable();
    GizmoDrawable(const GizmoDrawable& copy, osg::CopyOp op=osg::CopyOp::SHALLOW_COPY);

    enum Mode { NO_GIZMO=0, MOVE_GIZMO, ROTATE_GIZMO, SCALE_GIZMO };
    void setGizmoMode( Mode m, IGizmo::LOCATION loc=IGizmo::LOCATE_LOCAL );
    
    Mode getGizmoMode() const { return _mode; }
    IGizmo* getGizmoObject() { return _gizmo; }
    const IGizmo* getGizmoObject() const { return _gizmo; }
    
    void setTransform( osg::MatrixTransform* node );
    
    osg::MatrixTransform* getTransform() { return _transform.get(); }
    const osg::MatrixTransform* getTransform() const { return _transform.get(); }
    
    void setScreenSize( int w, int h );
    
    void applyTransform();
    
    META_Object( osg, GizmoDrawable );
    
    virtual void drawImplementation( osg::RenderInfo& renderInfo ) const;
    
protected:
    virtual ~GizmoDrawable() {}
    
    osg::observer_ptr<osg::MatrixTransform> _transform;
    IGizmo* _gizmo;
    float _editMatrix[16];
    int _screenSize[2];
    Mode _mode;
};

class MyTrackballManipulator : public osgGA::TrackballManipulator
{
public:
    bool handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa )
    {
        if ( !(ea.getModKeyMask()&osgGA::GUIEventAdapter::MODKEY_CTRL) )
            return false;  // force using CTRL when manipulating
        return osgGA::TrackballManipulator::handle(ea, aa);
    }
};

// int main( int argc, char** argv )
// {
//     osg::ref_ptr<osg::MatrixTransform> scene = new osg::MatrixTransform;
//     scene->addChild( osgDB::readNodeFile("/home/AD.EKUPD.COM/matthias.epple/trailer.3ds") );
    
//     osg::ref_ptr<GizmoDrawable> gizmo = new GizmoDrawable;
//     gizmo->setTransform( scene.get() );
//     gizmo->setGizmoMode( GizmoDrawable::MOVE_GIZMO );
    
//     osg::ref_ptr<osg::Geode> geode = new osg::Geode;
//     geode->addDrawable( gizmo.get() );
//     geode->setCullingActive( false );  // allow gizmo to always display
//     geode->getOrCreateStateSet()->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );  // always show at last
    
//     osg::ref_ptr<osg::MatrixTransform> root = new osg::MatrixTransform;
//     root->addChild( scene.get() );
//     root->addChild( geode.get() );
    
//     osgViewer::Viewer viewer;
//     viewer.setSceneData( root.get() );
//     viewer.setCameraManipulator( new MyTrackballManipulator );
//     viewer.addEventHandler( new osgGA::StateSetManipulator(viewer.getCamera()->getOrCreateStateSet()) );
//     viewer.addEventHandler( new osgViewer::StatsHandler );
//     viewer.addEventHandler( new osgViewer::WindowSizeHandler );
// 	viewer.realize();
    
//     osgViewer::GraphicsWindow* gw = dynamic_cast<osgViewer::GraphicsWindow*>( viewer.getCamera()->getGraphicsContext() );
//     if ( gw )
//     {
//         // Send window size for libGizmo to initialize
//         int x, y, w, h; gw->getWindowRectangle( x, y, w, h );
//         viewer.getEventQueue()->windowResize( x, y, w, h );
//     }
// 	return viewer.run();
// }
#endif