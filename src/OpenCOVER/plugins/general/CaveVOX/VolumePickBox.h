/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VOLUME_PICK_BOX_H_
#define _VOLUME_PICK_BOX_H_

// CUI:
#include <Interaction.H>
#include <PickBox.H>
#include <Marker.H>
#include <InputDevice.H>
#include <LogFile.H>
#include <CheckBox.H>
#include <TextureWidget.H>
#include <Panel.H>
#include <Measure.H>
#include <Rectangle.H>
#include <RadioButton.H>
#include <RadioGroup.H>
#include <Bar.H>
#include <Paintbrush.H>
#include <HeightFieldPickBox.H>

#include <vvvoldesc.h>

/**
    This class provides a volume object in a PickBox. The pick box
    provides a wireframe box around the volume, and it allows the volume
    to be moved around by the user.
*/
class VolumePickBox : public cui::PickBox, public cui::MarkerListener, public cui::CardListener, public cui::MeasureListener, public cui::RectangleListener, public cui::RadioGroupListener
{
public:
    enum PaintType
    {
        BOX,
        LINE,
        SPHERE
    };

protected:
    PaintType _pt;
    bool _markupMode; ///< true = markers visible
    bool _paintMode; ///< true = paintbrush visible
    cui::Interaction *_interaction;
    cui::InputDevice::CursorType _prevType; ///< pointer style when pointer entered box
    bool _settingPointerLength;
    bool _isNavigating; ///< true=button down and navigating data set
    bool _moveThresholdReached; ///< time difference to click big enough to move data set?
    bool _gazeSupport;
    bool _isVirvo;
    bool _isPainting;
    cui::LogFile *_logFile;
    cui::Marker::GeometryType _markerType;
    cui::Paintbrush::GeomType _paintType;
    osg::Vec3 _lastPoint;
    osg::Vec4 _color;
    int _lineSize;

    cui::Measure *_measure;
    cui::Rectangle *_rectangle;
    cui::Bar *_paintLine;
    cui::Panel *_diagramPanel;
    cui::TextureWidget *_diagTexture;
    cui::RadioGroup _radioGroup1;
    cui::RadioGroup _radioGroup2;
    cui::RadioButton *_intensityButton;
    cui::RadioButton *_histogramButton;
    cui::RadioButton *_lineButton;
    cui::RadioButton *_rectangleButton;
    cui::CheckBox *_redChannel;
    cui::CheckBox *_greenChannel;
    cui::CheckBox *_blueChannel;
    cui::CheckBox *_alphaChannel;

    osg::Image *_histoImage;
    osg::Image *_intensImage;

    cui::HeightFieldPickBox *_heightField;

    unsigned char _selChannel;

    void createDiagramPanel();
    void setChannelBoxesVisible();
    void setCheckedChannels(vvVolDesc::Channel, bool);
    unsigned char getCheckedChannels();

    virtual bool moveThresholdTest(osg::Matrix &, osg::Matrix &);
    virtual void processMoveInput(cui::InputDevice *);

public:
    std::vector<cui::Marker *> _markers;
    VolumePickBox(cui::Interaction *, osgDrawObj *, const osg::Vec3 &, const osg::Vec3 &,
                  const osg::Vec4 &, const osg::Vec4 &, const osg::Vec4 &);
    virtual ~VolumePickBox();
    virtual void cursorEnter(cui::InputDevice *);
    virtual void cursorLeave(cui::InputDevice *);
    virtual void cursorUpdate(cui::InputDevice *);
    virtual void trackballRotation(float, float, osg::Matrix &);
    virtual void setMarkupMode(bool);
    virtual bool getMarkupMode();
    virtual void setPaintMode(bool);
    virtual bool getPaintMode();
    virtual void setPaintType(PaintType);
    virtual PaintType getPaintType();
    virtual bool isVirvo()
    {
        return _isVirvo;
    }
    virtual std::vector<cui::Marker *> getMarkers()
    {
        cerr << "NumMarkers: " << _markers.size() << endl;
        return _markers;
    }
    virtual void addMarkerByHand(cui::Marker *);
    virtual void placeMarker(cui::Marker *);
    virtual void paint(int);
    virtual void paintLine();
    virtual osg::Vec3 getPointInVolume();
    virtual osg::Vec3 getVolume2Voxel(osg::Vec3);
    virtual void setColor(osg::Vec4);
    virtual void clear();
    virtual void addMarkerFromFile(cui::Marker *);
    virtual bool writeMarkerFile(const char *);
    virtual bool readMarkerFile(const char *);
    virtual void removeMarker(cui::Marker *);
    virtual int removeRandMarkersUntil(int);
    virtual int removeAllMarkers();
    virtual void scaleMarkers(float);
    virtual void buttonEvent(cui::InputDevice *, int);
    virtual void setAllMarkerSizes(float);
    virtual void setGazeSupport(bool);
    virtual void setLogFile(cui::LogFile *);
    virtual void getMarkersFromLog(const char *);
    virtual void setMarkerType(cui::Marker::GeometryType);
    virtual cui::Marker::GeometryType getMarkerType();
    virtual void setLineSize(int);

    // From Marker:
    virtual void markerEvent(cui::Marker *, int, int);

    // From CardListener:
    virtual bool cardButtonEvent(cui::Card *, int, int);
    virtual bool cardCursorUpdate(cui::Card *, cui::InputDevice *);

    // From MeasureListener:
    virtual void measureUpdate();

    // From RectangleListener::
    virtual void rectangleUpdate();

    // From RadioGroupListener:
    virtual bool radioGroupStatusChanged(cui::RadioGroup *);

    /*
    @param flag: true:  show _measure
                 false: hide _measure
  */
    void setMeasureVisible(bool flag);

    bool getMeasureVisible();

    /*
    @param flag: true:  show _rectangle
                 false: hide _rectangle
  */
    void setRectangleVisible(bool flag);

    bool getRectangleVisible();

    /*
    @param flag: true:  show _diagramPanel
                 false: hide _diagramPanel
  */
    void setDiagramPanelVisible(bool flag);

    /*
    @param flag: true:  show _histoImage;
                 false: show _intensImage;
  */
    void showDiagram();
};
#endif
