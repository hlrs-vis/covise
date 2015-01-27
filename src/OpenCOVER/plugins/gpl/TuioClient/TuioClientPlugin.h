// TuioClientPlugin.h

#ifndef TUIO_CLIENT_PLUGIN_H
#define TUIO_CLIENT_PLUGIN_H

#include <cover/coVRPlugin.h>

#include <TuioClient.h>

#include <string>

class TuioClientPlugin
    : public opencover::coVRPlugin,
      public TUIO::TuioListener
{
public:
    TuioClientPlugin();

    virtual ~TuioClientPlugin();

    //
    // coVRPlugin interface
    //

    //! this function is called when COVER is up and running and the plugin is initialized
    virtual bool init();

    //! reimplement to do early cleanup work and return false to prevent unloading
    virtual bool destroy();

    //
    // TuioListener interface
    //

    //! This callback method is invoked by the TuioClient when a new TuioObject is added to the session.
    virtual void addTuioObject(TUIO::TuioObject *tobj);

    //! This callback method is invoked by the TuioClient when an existing TuioObject is updated during the session.
    virtual void updateTuioObject(TUIO::TuioObject *tobj);

    //! This callback method is invoked by the TuioClient when an existing TuioObject is removed from the session.
    virtual void removeTuioObject(TUIO::TuioObject *tobj);

    //! This callback method is invoked by the TuioClient when a new TuioCursor is added to the session.
    virtual void addTuioCursor(TUIO::TuioCursor *tcur);

    //! This callback method is invoked by the TuioClient when an existing TuioCursor is updated during the session.
    virtual void updateTuioCursor(TUIO::TuioCursor *tcur);

    //! This callback method is invoked by the TuioClient when an existing TuioCursor is removed from the session.
    virtual void removeTuioCursor(TUIO::TuioCursor *tcur);

    //! This callback method is invoked by the TuioClient to mark the end of a received TUIO message bundle.
    virtual void refresh(TUIO::TuioTime ftime);

private:
    TUIO::TuioClient *tuioClient;
    std::string destination;
};
#endif
