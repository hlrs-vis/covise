set Help(COVERConfig,STEREO_SEPARATION) {separation between left and right eye in mm
float
default: 64
don't mix up STEREO_SEPARATION and STEREO_SEP !
}
set Help(COVERConfig,FORCE_FEEDBACK_MODE) {for phantom force feedback device
contact woessner@hlrs.de for details
}
set Help(COVERConfig,MODELFILE) {load a geometry file
supported are all formats which are also supported by performer
other possibility: cover <modelfile/modelfile>
or use module PerformerScene in a COVISE map 

}
set Help(COVERConfig) {This is the help for the
COVERConfig section
}
set Help(COVERConfig,MAX_FORCE) {for phantom force feedback device
contact woessner@hlrs.de for details
}
set Help(COVERConfig,COLLIDE) {enable/disable collision detection
default OFF
}
set Help(TrackerConfig,HANDSENSOR_ORIENTATION) {Help for HANDSENSOR_ORIENTATION




}
set Help(COVERConfig,ANTIALIAS) {enable/disable antialiasing
boolean
default: ON
antialiasing is automatically disabled if not supported 
by the graphics board
check if you have a visual which supports antialiasing and quadbuffer stereo
use findvis on sgi

}
set Help(COVERConfig,NEAR) {near plane of the opengl viewing frustum
default: 10.0
}
set Help(COVERConfig,FORCE_SCALE) {for phantom force feedback device
contact woessner@hlrs.de for details
}
set Help(COVERConfig,stepSize) {step size (lengths of a step, important for stairs
default: 400
}
set Help(Hosts) {For each section 
there may exist a hostspecific version.
Use this dialogue to add a new host so that you can 
configure several sections for it.



}
set Help(COVERConfig,TextureQuality) {texture quality for vrml files
default: High
}
set Help(COVERConfig,FOOTER_MESSAGE) {this message is drawn on the bottom
}
set Help(COVERConfig,SAVE_FILE) {store the screnegraph as pfb to this file name 
when "store scenegraph" is selected from the pinboard
default: /var/tmp/COVER.pfb
}
set Help(TrackerConfig,HANDSENSOR_OFFSET) {   
    This is the position of the Handsensor
    The coordinates are given in cm
   
}
set Help(COVERConfig,NUM_PIPES) {This is the number of hard pipes i.e. graphics cards 
in your computer.
Pipes are configured in section PipeConfig



}
set Help(COVERConfig,BACKGROUND) {background color
3 floats (r g b)
default 0.0 0.0 0.0 (black)
}
set Help(COVERConfig,FREEZE) {start value for headtracking
default ON (this means headtracking is off)
}
set Help(COVERConfig,NOTIFY) {NOTIFY sets the threshold for notification.  A notification must
have a level less than or equal to the threshold for the default handler
to print a message.  The notification handler itself is invoked
regardless of the notification level.  The levels are in decreasing
severity:
          PFNFY_ALWAYS
          PFNFY_FATAL
          PFNFY_WARN
          PFNFY_NOTICE
          PFNFY_INFO
          PFNFY_DEBUG
          PFNFY_FP_DEBUG.



}
set Help(COVERConfig,SyncInterval) {synchronisation time intervall in seconds
default 0 (which means immediatly)
}
set Help(COVERConfig,STEREO) {enable/disable active (quadbuffer) stereo
boolean
default: ON
stereo is automatically disabled if not supported by the graphics board
}
set Help(COVERConfig,COORD_AXIS) {show/hide coordinate axis for world, hand and viewer coordinate system
default: OFF
}
set Help(COVERConfig,SPOTLIGHT) {enable/disable specular color of lights
default: off
}
set Help(UIConfig,AutoSaveTime) { Seconds between two automatic saves
}
set Help(COVERConfig,DEBUG_LEVEL) {COVER debug level
Number from 0 - 6
default: 0
}
set Help(COVERConfig,NO_SURROUND) {enable/disable surround sound for vrml files
default:ON
enable surround sound only if you have a surround amplifier
}
set Help(COVERConfig,TextureMode) {defines blending of object colors and texture colors in vrml files
default: MODULATE


}
set Help(COVERConfig,MENU_ORIENTATION) {Orientation of the menu
3 floats (euler angles, h, p, r)
default 0 0 0

}
set Help(COVERConfig,SYNC_MODE) {synchronisation mode for collaborative working
default: OFF
}
set Help(COVERConfig,SCENESIZE) {size in mm to which a scene is scaled
default 300
good choose is the smaller dimension of the screen
}
set Help(COVERConfig,ARENA_SIZE) { Size of shared arena used for multiprocessing in bytes
default: 250000000
if you need larger sizes make sure you have enough disk space
}
set Help(ButtonConfig,SERIAL_PORT) {Der serialle Port des Button devices
}
set Help(COVERConfig,SNAP) {enable/disable snapping at start time
default OFF
snapping is only used by cuttingsurfaces and isosurfaces

}
set Help(COVERConfig,AUTO_WIREFRAME) {
}
set Help(COVERConfig,LookAndFeel) {Look of interactors 
default: green
icon directory $COVISEDIR/icons/$LookAndFeel
default icon directory: $COVISEDIR/icons
}
set Help(TrackerConfig,TRANSMITTER_ORIENTATION) {





}
set Help(COVERConfig,TWOSIDE) {enable/disable twosided lighting
default ON
twosided lighting is needed to display faces whith normals pointng 
away from the viewer
}
set Help(COVERConfig,BUTTON_SYSTEM) {button system type
configure a button system only if the buttons are not handled 
by the tracking system
The button system is configured in section ButtonConfig
}
set Help(COVERConfig,STIPPLE) {stipple stereo, displays right and left eye on even/uneven pixles 
in x direction
boolean
default OFF
this stereo mode works only with the Dresden3D Display
}
set Help(COVERConfig,NUM_SCREENS) {Number of projections screens (better: number of projectors)
default 1
for active stereo you have one projector per physical screen
for passive stereo you have two projectors per physical screen
screens are configured in ScreenConfig     
    NUM_SCREENS                 1
}
set Help(COVERConfig,MULTIPROCESS) {Performer multiprocessing
default: ON
disable on Linux with Performer<2.5 

}
set Help(COVERConfig,PIPE_LOCKING) {synchronisation of swapbuffer of different pipes
default: CHANNEL 
WINDOW uses pfPipeWindow::setSwapGroup
CHANNEL uses pfChannel::setShare(PFCHAN_SWAPBUFFERS)
CHANNEL_HW uses pfChannel::setShare(PFCHAN_SWAPBUFFERS)
for CHANNEL_HW the SwapReady cable needs to be connected
}
set Help(COVERConfig,WELCOME_MESSAGE) {This message is drawn when COVER starts
default: nothing is printed
}
set Help(COVERConfig,VISUAL_ID) {select framebuffer configuration by visual id
default: not set, framebuffer configuration through STEREO and ANTIALIAS
on sgi you get the available visuals with the command 
'findvis -display :0.0'
select a visual which support doublebuffer (db), zbuffer (Z), rgba (RGBA), 
stereo (stereo) and multisampling (S)
}
set Help(COVERConfig,NOSEPARATION) {set separation to zero
boolean
default: OFF 
}
set Help(COVERConfig,MONO_COMMAND) {command which switches to the mono videoformat
default: not videoformat switching
usually the stereo videoformat or combination is already loaded
and you should't change it

}
set Help(COVERConfig,stateFixLevel) {defines to which node in a vrml file a state fix callback is attached
state fix callbacks are a workaround for a bug in Performer
default: 100
stateFixLevel<0 no callback
stateFixLevel=1: vrml root dcs
stateFixLevel=3: group nodes
stateFixLevel>5: all nodes
contact woessner@hlrs.de for details

}
set Help(COVERConfig,SCALE_ALL) {automatic scale scene to the size defined with SCENESIZE if new objects 
are added to the scene
default: ON
		
}
set Help(COVERConfig,FAR) {far plane of the opengl viewing frustum
default: near+10000000

}
set Help(COVERConfig,FORCE_FEEDBACK) {for phantom force feedback device
contact woessner@hlrs.de for details
}
set Help(COVERConfig,NoPreload) {avoid preload of switch nodes in vrml
default: false
}
set Help(COVERConfig,VIEWER_POSITION) {viewer position at starttime if headtracking is off (FREEZE ON)
3 floats (x, y, z position of viewer)
default: x=0, y=-450 z=0
}
set Help(TrackerConfig,HEADSENSOR_OFFSET) {qweqwewqe

}
set Help(COVERConfig,TRACKING_SYSTEM) {tracking system type
default: SPACEPOINTER
the tracking system is configured with the sections TrackerConfig 
and <system>.config , for example PolhemusConfig
 
}
set Help(COVERConfig,MENU_SIZE) {menu size
default 1
}
set Help(COVERConfig,DebugSound) {enable/disable debug prints for sound in vrml files
default false
}
set Help(COVERConfig,STEREO_COMMAND) {command which switches to the stereo videoformat
default: not videoformat switching
usually the stereo videoformat or combination is already loaded
and you should't change it


}
set Help(COVERConfig,MAX_HEAP_BYTES) {size of java script heap in MB ?
default: 8
}
set Help(COVERConfig,LIGHT1) {two additional lights
4 floats x y z w
position: x y z  and w=1 local, w=0 infinite
default: no extra lights
}
set Help(COVERConfig,MENU_POSITION) {Position of the menu
3 floats for x y z 
the unit is [mm in the rus cube]
default psoition is 0 0 0
}
set Help(COVERConfig,STEREO_SEP) {enable/disable stereo separation at starttime
default OFF
don't mix up STEREO_SEP and STEREO_SEPARATION, the first one
is the start mode for a button in the pinboard which switches between
separation=STEREO_SEPARATION and 0
}
set Help(COVERConfig,FPS) {print out the number of frames per second every frame
default OFF
}
set Help(COVERConfig,DRIVE_SPEED) {navigation speed for fly, drive and walk
3 floats (min, max, val)
default: 0.0 30.0 1.0
}
set Help(COVERConfig,NUM_WINDOWS) {Number of windows
default 1
usually one window per pipe      
windows are configured in section WindowConfig
}
set Help(COVERConfig,LIGHT2) {two additional lights
4 floats x y z w
position: x y z  and w=1 local, w=0 infinite
default: no extra lights
}
set Help(COVERConfig,floorHeight) {z position of the floor in relation to the world coordinate system
default:-1250
used for walk mode collision detection
}
set Help(COVERConfig,NAVIGATION_MODE) {navigation mode at starttime
#-- default: OFF

}
set Help(COVERConfig,ANIM_SPEED) {start value for animation speed slider
3 floats (min, max, val)
default:  0.0 30.0 1.0
}
set Help(COVERConfig,LOD_SCALE) {Multiply all LOD scales with this factor
default: 1.0

}
set Help(TrackerConfig,DEBUG_TRACKING) { 
    For debugging the data given by the tracker
    if RAW ist used then
    if APP is used then
 }
set Help(COVERConfig,CORRECT_MATRIX) {enable for environment maps
default ON

}
set Help(COVERConfig,MODELPATH) {load a geometry file
supported are all formats which are also supported by performer
other possibility: cover <modelfile/modelfile>
or use module PerformerScene in a COVISE map
}
