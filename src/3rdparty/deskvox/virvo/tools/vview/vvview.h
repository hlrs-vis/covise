//****************************************************************************
// Project:         Virvo (Virtual Reality Volume Renderer)
// Copyright:       (c) 1999-2004 Jurgen P. Schulze. All rights reserved.
// Author's E-Mail: schulze@cs.brown.edu
// Affiliation:     Brown University, Department of Computer Science
//****************************************************************************

#ifndef VVVIEW_H
#define VVVIEW_H

/**
 * Virvo File Viewer main class.
 * The Virvo File Viewer is a quick alternative to the Java based VEdit
 * environment. It can display all Virvo file types using any implemented
 * algorithm, but it has only limited information and transfer function edit
 * capabilites.<P>
 * Usage:
 * <UL>
 *   <LI>Accepts the volume filename as a command line argument
 *   <LI>Mouse moved while left button pressed: rotate
 *   <LI>Mouse moved while middle button pressed: translate
 *   <LI>Mouse moved while right button pressed and menus are off: scale
 * </UL>
 *
 *  This program supports the following macro definitions at compile time:
 * <DL>
 *   <DT>VV_DICOM_SUPPORT</DT>
 *   <DD>If defined, the Papyrus library is used and DICOM files can be read.</DD>
 *   <DT>VV_VOLUMIZER_SUPPORT</DT>
 *   <DD>If defined, SGI Volumizer is supported for rendering (only on SGIs).</DD>
 * </DL>
 *
 * @author Juergen Schulze (schulze@cs.brown.de)
 */

class vvOffscreenBuffer;
class vvStopwatch;
class vvObjView;
class vvBrick;

#include <vector>

#include <virvo/vvvecmath.h>
#include <virvo/vvimage.h>
#include <virvo/vvclock.h>
#include <virvo/vvrenderer.h>
#include <virvo/vvrendererfactory.h>

class vvView
{
  private:
    /// Mouse buttons (to be or'ed for multiple buttons pressed simultaneously)
    enum
    {
      NO_BUTTON     = 0,                       ///< no button pressed
      LEFT_BUTTON   = 1,                       ///< left button pressed
      MIDDLE_BUTTON = 2,                       ///< middle button pressed
      RIGHT_BUTTON  = 4                        ///< right button pressed
    };
    enum                                        ///  Timer callback types
    {
      ANIMATION_TIMER = 0,                     ///< volume animation timer callback
      ROTATION_TIMER  = 1,                     ///< rotation animation timer callback
      BENCHMARK_TIMER  = 2                     ///< benchmark timer callback
    };
    /// Clipping edit mode
    enum
    {
      PLANE_X = 0,                              ///< rotate plane normal along x-axis
      PLANE_Y,                                  ///< rotate plane normal along y-axis
      PLANE_Z,                                  ///< rotate plane normal along z-axis
      PLANE_NEG,                                ///< move plane along negative normal dir
      PLANE_POS                                 ///< move plane along positive normal dir
    };
    /// Remote rendering type
    enum RemoteType
    {
      RR_NONE = 0,
      RR_COMPARISON,
      RR_IMAGE,
      RR_IBR,
      RR_PARBRICK
    };
    static const int ROT_TIMER_DELAY;           ///< rotation timer delay in milliseconds
    static const int DEFAULTSIZE;               ///< default window size (width and height) in pixels
    static const float OBJ_SIZE;                ///< default object size
    static const int DEFAULT_PORT;              ///< default port for socket connections
    static vvView* ds;                          ///< one instance of VView is always present
    vvObjView* ov;                              ///< the current view on the object
    vvRenderer* renderer;                       ///< rendering engine
    vvRenderState renderState;                  ///< renderer state
    vvVolDesc* vd;                              ///< volume description
    char* filename;                             ///< volume file name
    int   window;                               ///< GLUT window handle
    int   winWidth, winHeight;                  ///< window size in pixels
    int   pressedButton;                        ///< ID of currently pressed button
    int   lastX, lastY;                         ///< previous mouse coordinates
    int   curX, curY;                           ///< current mouse coordinates
    int   x1,y1,x2,y2;                          ///< mouse coordinates for auto-rotation
    int   lastWidth, lastHeight;                ///< last window size
    int   lastPosX, lastPosY;                   ///< last window position
    bool  emptySpaceLeapingMode;                ///< true = bricks invisible due to current transfer function aren't rendered
    bool  earlyRayTermination;                  ///< true = don't compute invisible fragments
    bool  perspectiveMode;                      ///< true = perspective projection
    bool  boundariesMode;                       ///< true = display boundaries
    bool  orientationMode;                      ///< true = display axis orientation
    bool  fpsMode;                              ///< true = display fps
    bool  paletteMode;                          ///< true = display transfer function palette
    int   stereoMode;                           ///< 0=mono, 1=active stereo, 2=passive stereo (views side by side)
    bool  activeStereoCapable;                  ///< true = hardware is active stereo capable
    bool  tryQuadBuffer;                        ///< true = try to request a quad buffered visual
    virvo::tex_filter_mode  filter_mode;        ///< true = linear interpolation in slices
    bool  warpInterpolMode;                     ///< true = linear interpolation during warp (shear-warp)
    bool  preintMode;                           ///< true = use pre-integration
    bool  opCorrMode;                           ///< true = do opacity correction
    bool  gammaMode;                            ///< true = do gamma correction
    int   mipMode;                              ///< 1 = maximum intensity projection, 2=minimum i.p.
    bool  fullscreenMode;                       ///< true = full screen mode enabled
    bool  timingMode;                           ///< true = display rendering times in text window
    bool  menuEnabled;                          ///< true = popup menu is enabled
    int   mainMenu;                             ///< ID of main menu
    bool  animating;                            ///< true = animation mode on
    bool  rotating;                             ///< true = rotation mode on
    bool  rotationMode;                         ///< true = auto-rotation possible
    size_t frame;                               ///< current animation frame
    std::string currentRenderer;                ///< current renderer/rendering geometry
    vvRendererFactory::Options currentOptions;  ///< current options/voxel type
    vvVector3 bgColor;                          ///< background color (R,G,B in [0..1])
    float draftQuality;                         ///< current draft mode rendering quality (>0)
    float highQuality;                          ///< current high quality mode rendering quality (>0)
    bool  hqMode;                               ///< true = high quality mode on, false = draft quality
    bool  refinement;                           ///< true = use high/draft quality modes, false = always use draft mode
    const char* onOff[2];                       ///< strings for "on" and "off"
    vvVector3 pos;                              ///< volume position in object space
    float animSpeed;                            ///< time per animation frame
    bool  iconMode;                             ///< true=display file icon
    int bricks;                                 ///< num bricks (serbrickrend and parbrickrend)
    std::vector<std::string> displays;          ///< for parbrickrend
    int isectType;
    bool useOffscreenBuffer;                    ///< render to an offscreen buffer. Mandatory for setting buffer precision
    bool useHeadLight;                          ///< toggle head light
    int  bufferPrecision;                       ///< 8 or 32 bit. Higher res can minimize rounding error during slicing
    RemoteType  rrMode;                         ///< memory remote rendering mode
    int ibrPrecision;                           ///< Precision of depth buffer in image based (remote-)rendering mode
    vvRenderState::IbrMode          ibrMode;    ///< interruption mode for depth-calculation
    bool sync;                                  ///< synchronous ibr mode
    int codec;                                  ///< code type/codec for images sent over the network
    vvOffscreenBuffer* clipBuffer;              ///< used for clipping test code
    GLfloat* framebufferDump;
    std::vector<std::string> servers;
    std::vector<int> ports;
    std::vector<vvSocket*> sockets;
    bool benchmark;                             ///< don't run interactively, just perform timed rendering and exit
    std::vector<std::string> serverFileNames;   ///< a list with file names where remote servers can find the appropriate volume data
    const char* testSuiteFileName;
    bool showBricks;                            ///< show brick outlines when brick renderer is used
    bool recordMode;                            ///< mode where camera motion is saved to file
    bool playMode;                              ///< mode where camera motion is played from file
    FILE* matrixFile;                           ///< Modelview matrices recorded in record mode
    vvStopwatch stopWatch;                      ///< used in record mode to store time along with each frame
    bool roiEnabled;                            ///< mode where probe is shown and can be moved via arrow keys
    bool sphericalROI;                          ///< use sphere instead of cube
    bool clipMode;                              ///< clip mode [on/off]
    bool clipPerimeter;                         ///< draw clip perimeter
    bool clipEditMode;                          ///< edit clip plane using keyboard
    float mvScale;                              ///< scale factor for the mv matrix to view the whole volume
    vvVector3 planeRot;                         ///< rotation of clipping plane normal
    bool showBt;                                ///< Show backtrace if execution stopped due to OpenGL error
    bool ibrValidation;

  public:
    vvView();
    ~vvView();
    int run(int, char**);
    static void cleanup();

  private:
    static void reshapeCallback(int, int);
    static void displayCallback();
    static void buttonCallback(int, int, int, int);
    static void motionCallback(int, int);
    static void keyboardCallback(unsigned char, int, int);
    static void specialCallback(int, int, int);
    static void timerCallback(int);
    static void mainMenuCallback(int);
    static void rendererMenuCallback(int);
    static void voxelMenuCallback(int);
    static void optionsMenuCallback(int);
    static void transferMenuCallback(int);
    static void animMenuCallback(int);
    static void roiMenuCallback(int);
    static void clipMenuCallback(int);
    static void viewMenuCallback(int);
    static void runTest();
    static double performanceTest();
    static void printProfilingInfo(const int testNr = 1, const int testCnt = 1);
    static void printProfilingResult(vvStopwatch* totalTime, const int framesRendered);
    static void printROIMessage();
    void setAnimationFrame(ssize_t);
    void initGraphics(int argc, char *argv[]);
    void createMenus();
    void createRenderer(std::string renderertype, const vvRendererFactory::Options &opt,
                        size_t maxBrickSizeX = 64, size_t maxBrickSizeY = 64, size_t maxBrickSizeZ = 64);
    void applyRendererParameters();
    void setProjectionMode(bool);
    void renderMotion() const;
    void editClipPlane(int command, float val);
    void displayHelpInfo();
    bool parseCommandLine(int argc, char *argv[]);
    void mainLoop(int argc, char *argv[]);
};
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
