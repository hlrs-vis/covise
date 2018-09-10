************************************
* VOX - Volume Exploration Utility *
************************************


Author
------

Jurgen P. Schulze
University of California, San Diego
Calit2
9500 Gilman Dr.
La Jolla, CA 92093-0436
USA

URL: http://www.calit2.net/~jschulze/


Short Description
-----------------

DeskVOX is a real-time visualization tool for 3D data sets like image 
stacks from CT or MRI scanners, or confocal microscopes. 
It has an easy to use GUI and runs under Windows or Linux.

The package contains two software tools for interactive direct volume rendering:

- DeskVOX is a volume viewer with a graphical user interface.
- VConv is a command line volume file conversion tool.

Volume data sets are rendered with graphics hardware accelerated algorithms
based on 2D and 3D textures.


Detailed Description
--------------------

DeskVOX stands for "VOlume eXplorer for the Desktop". Its development started in 1999 while the author was a graduate student at the University of Stuttgart in Germany. The goal of DeskVOX is to provide interactive direct volume rendering on desktop computers. DeskVOX has a sibling for virtual environments like CAVEs or 3D stereo walls: CaveVOX. CaveVOX shares the rendering library with DeskVOX, but uses a different user interface. The software development has continued after the author moved to Brown University in 2003.

A typical use case is to visualize images from a CT or MRI scan. DeskVOX accepts TIFF files as a source, which will be converted to a 3D volume file, which can be viewed interactively. Several visual features support the exploration of such data sets: zoom, pan, a clipping plane, a region of interest, viewer settings, and stereo modes.

One distinguishing feature of DeskVOX is its capability to display multi-channel data sets, for instance up to four channel confocal data sets. Another uncommon feature is that not only data sets with 8 bits per voxel are supported, but also 16 bit integers and 32 bit floating point values. Furthermore, several 3D stereo modes are available.

The volume rendering routines require a graphics card with hardware based texturing acceleration. Most graphics cards built after 2001 (GeForce3, Radeon, etc.) have this capability. Newer graphics cards and those with more texture memory will achieve higher performance and will render bigger data sets.

DeskVOX comes with the command line utility VConv, which converts between a variety of image and volume data formats. It also converts volume data sets between different data formats, or it resamples data sets to different sizes.

The GUI of DeskVOX uses FOX Toolkit, which is an open-source, cross-platform library for user interface development. The Windows version of DeskVOX auto-installs thanks to the Nullsoft Install System (NSIS).


System Requirements
-------------------

- Required: A Windows or Linux based computer.
- Recommended: Nvidia GeForce3 graphics card or higher, or equivalent ATI graphics card.
- Optimum: Nvidia GeforceFX graphics card or better.


Program Usage
-------------

- DeskVOX: This is a GUI based program and can be started from the 
           Windows Explorer. 
           Mouse button assignment in output window:
           - Left button: rotate
           - Middle button: pan
           - Right button: scale

- VConv:  This is a command line tool. 
          'vconv -help' gives detailed information about its parameters.


Tip when running under Windows
------------------------------

Volume files can be passed to the program by drag-and-drop to the program icon. 
If you create an entry in the Explorer's View/Options/FileType list for xvf and rvf files, 
you can display volume files by double-clicking on them.


Supported Volume File Formats
-----------------------------

.dat: Raw volume data without any header. The data can have 1,2,3, or 4 byte per voxel.
      The program tries to find the volume dimensions automatically, but you can help by giving
      them in the file name, e.g. "cthead256x256x64.dat" for a 256 x 256 x 64 voxels 
      dataset. Data order: starting from top-left-front, going right first, then down, then back.
      All bytes of each voxel must be stored consecutively, starting with the most
      significant byte or in RGBA order, depending on the data type.

.rvf: Virvo format for 'raw volume data'. 
      This format can easily be created by hand from any voxel data array on disk 
      by just adding the appropriate header.
      Header: 3 x 2 Bytes (big endian) = volume width/height/depth (voxels)
              Example for a 256x128x127 volume (hex values): 01 00 00 80 00 7F 
      Volume data format: 8 bit per voxel
      Data order: starting from top-left-front, going right first, then down, then back

.xvf: Virvo format for 'extended volume data'.
      This format can handle more complex data but is still rather easy to describe.
      It features: multiple volume datasets (time steps) in one file, and the 
      storage of multiple transfer functions. 
      The data format is not restricted to 8 bit per voxel but can be 8, 16, 24 or 32
      bits per voxel.
      Here is the file format specification (byte order: most significant first):
      
     Header: 
       Offset Bytes Data Type       Description
     ---------------------------------------------------------------
          0   9     char            file ID string: "VIRVO-XVF"
          9   2     unsigned short  offset to beginning of data area, from top of file [bytes]
         11   2 x 4 unsigned int    width and height of volume [voxels]
         19   4     unsigned int    number of slices per time step
         23   4     unsigned int    number of frames in volume animation (time steps)
         27   1     unsigned char   bytes per voxel (for details see vvvoldesc.h)
         28   3 x 4 float           real world voxel size (width, height, depth) [mm]
         40   4     float           length of a time step in the volume animation [seconds]
         44   2 x 4 float           physical data range covered by voxel data (minimum, maximum)
         52   3 x 4 float           real world location of volume center (x,y,z) [mm]
         64   1     unsigned char   storage type (for details see vvvoldesc.h)                             
         65   1     unsigned char   compression type (0=uncompressed, 1=RLE)
         66   2     unsigned short  number of transfer functions
         68   2     unsigned short  type of transfer function: 0 = 4 x 256 Byte, 
                                    1 = list of control pins
         70   2     unsigned int    icon size: width (=height) [pixels]
         
     Data area:           
       Data starts at "offset to data area".
       Voxel order: voxel at top left front first, then to right, 
       then to bottom, then to back, then frames. All bytes of each voxel
       are stored successively.
       In RLE encoding mode, a 4 byte big endian value precedes each frame, 
       telling the number of RLE encoded bytes that will follow. If this 
       value is zero, the frame is stored without encoding.
    
     Now follow the transfer functions. 
     Each transfer function of type 0 consists of:
     - Zero terminated description string
     - Transfer function data in RGBA format: 
       First all R's, then all G's, etc.
       Each R/G/B/A entry is coded as one unsigned byte.
       The table length depends on the number of bits per voxel: 
        8 bits per voxel => 4 * 256 bytes (RGBA)
       16 bits per voxel => 4 * 4096 bytes (RGBA)
       24 bits per voxel => 3 * 256 bytes (alpha conversion for RGB)
       32 bits per voxel => 1 * 256 bytes (alpha conversion for density)
     - Transfer function in pin format:
       Each pin consists of 9 float values. The list is terminated by
       a set of -1.0 values.
    
     Hints: 
       The big endian hexadecimal representations of some important floating point values are:
        1.0 = 3F 80 00 00
       -1.0 = BF 80 00 00
      
.avf: Virvo format for 'ASCII volume data'.
   The files are expected to be in ASCII format. They consist of a header
   and a data section:

 Header: 

   The header consists of several identifier and value pairs to specify 
   the data format. Identifier and a value are separated by whitespace.
   This file format cannot store transfer functions.
   Unix-style comments starting with '#' are permitted.

   The following abbreviations are used:  
   <int>            for integer values
   <float>          for floating point values

   The following lines are required:
   WIDTH <int>      the width of the volume [voxels]
   HEIGHT <int>     the height of the volume [voxels]
   SLICES <int>     the number of slices in the volume [voxels]
   
   The following lines are optional. 
   If they are missing, default values are used:
   FRAMES <int>     the number of data sets contained in the file
                    (default: 1)
   MIN <float>      the minimum data value, smaller values will be constrained
                    to this value (default: 0.0)
   MAX <float>      the maximum data value, larger values will be constrained
                    to this value (default: 1.0)
   XDIST <float>    the sample distance in x direction (-> width) [mm] 
                    (default: 1.0)
   YDIST <float>    the sample distance in y direction (-> height) [mm]
                    (default: 1.0)
   ZDIST <float>    the sample distance in z direction (-> slices) [mm]
                    (default: 1.0)
   TIME <float>     the length of each time step for transient data [s]
                    (default: 1.0)
   BPC <int>        bytes per channel (1=8bit, 2=16bit, 4=float)
   CHANNELS <int>   number of data channels per voxel
   XPOS <float>     x position of data set center [mm] (default: 0.0)
   YPOS <float>     y position of data set center [mm] (default: 0.0)
   ZPOS <float>     z position of data set center [mm] (default: 0.0)

 Voxel data:         
   
   The voxel data starts right after the header. The data values
   are separated by whitespace and/or end-of-line characters.
   float and integer values are accepted.
   Voxel order: voxel at top left front first, then to right, 
   then to bottom, then to back, then frames. 
   All channels of each voxel are stored consecutively (interleaved).

 Sample file:

 WIDTH    4
 HEIGHT   3
 SLICES   2
 FRAMES   1
 MIN      0.0
 MAX      1.0
 XDIST    1.0
 YDIST    1.0
 ZDIST    1.0
 XPOS     0.0
 YPOS     0.0
 ZPOS     0.0
 TIME     1.0
 BPC      4
 CHANNELS 1
 0.9 0.9 0.9 0.9
 0.9 0.2 0.3 0.9
 0.9 0.2 0.4 0.9
 0.8 0.8 0.8 0.8
 0.8 0.1 0.1 0.8
 0.8 0.0 0.0 0.8

.tif, .tiff: 3D TIF File format. Virvo can only read files of this format, 
  it cannot write them.


Supported 2D Slice File Formats
-------------------------------

.rgb:                 SGI RGB image file (8 bit grayscale only)
.fre, .fro:           Visible Human CT data file
.pd, .t1, .t2, .loc:  Visible Human MRI file
.raw:                 Visible Human RGB (anatomy) file
.pgm:                 Portable Graymap file (binary version only)
.dcm:                 DICOM image file

To save the current Virvo volume to disk you can use the 'Save Volume Data' 
menu item and give the filename the desired name. The saved format depends
on the given filename extension. When loading, the file format is recognized
by the given extension again.


Movie Scripts
-------------

DeskVOX allows the user to generate movies with changing camera 
positions by loading ASCII movie script files (.vms).

The following movie script commands are recognized:

trans AXIS DIST
  Translates the data set by DIST in the AXIS axis.
  AXIS can be x, y, or z.

rot AXIS ANGLE
  Rotates the data set by ANGLE degrees about the AXIS axis.
  AXIS can be x, y, or z.
  
scale FACTOR
  Scales the data set by a factor of FACTOR. A value of 1.0 means no scaling.
  Values greater than 1.0 enlarge the data set, values smaller than 1.0 
  make it smaller.

timestep INDEX
  Display time step number INDEX in a volume animation. 
  The first time step has index 0.

nextstep
  Display the next time step of a volume animation. 
  Will switch to the first step after the last.  

prevstep
  Display the previous time step in a volume animation. 
  Will switch to the last step after the first.

setpeak POS WIDTH
  Set a peak pin with WIDTH width [0..1] to POS [0..1].
  This will overwrite all previously defined alpha pins.

movepeak DISTANCE
  Move alpha peak by DISTANCE. The total alpha range has extension 1.0,
  so a DISTANCE value of 0.1 would move the peak by 1/10th of the
  value range to the right.

setquality QUALITY
  Set rendering quality. 0 is worst, 1 is default, higher is better

changequality RELATIVE_QUALITY
  Changes the quality setting by a relative value. Quality value cannot
  get smaller than zero.
  
setclip X Y Z POS
  Define and enable a clipping plane. Use X,Y,Z,POS=0 to disable.

moveclip DX DY DZ DPOS
  Move clipping plane relative to current position.

setclipparam SINGLE OPAQUE PERIMETER
  Set clipping plane parameters: 
    SINGLE: 1=single slice, 0=cutting plane
    OPAQUE: 1=if single slice then make opaque, 0=use transfer function settings for slice
    PERIMETER: 1=show clipping plane perimeter, 0=don't show perimeter

show
  Displays the data set using the current settings.
  

Here is an example movie script file:

scale 1.2       # scale object by factor 1.2
rot x 20        # rotate 20 degrees about the x axis
rot y 25        # rotate 25 degrees about y axis
timestep 0      # switch to first time step
show            # display dataset
timestep 1      # switch to second time step
show            # display dataset
repeat 10       # repeat the following 10 times
  rot z 2       # rotate 2 degrees about z axis
  rot x 1       # rotate 1 degree about x axis
  show          # display dataset
endrep          # terminate repeat loop
rot z 10        # rotate 10 degrees about z axis
show            # display dataset
setpeak 0.0 0.1 # define peak at the lowest scalar value with width 0.1
show            # display dataset
repeat 5        # repeat the following 5 times
  movepeak 0.1  # move peak to the right by 1/10th of the scalar value range
  show          # display dataset
endrep          # terminate repeat loop


Known bugs
----------

- ARToolkit mode: when icon is unchecked, system crashes


Troubleshooting
---------------

 

Versions History
----------------

Nr.   Date      Details
------------------------------------------------------------------------------
1.0   99-11-11
1.1   99-11-17  released for SC99
1.2   99-11-18  released for SC99
1.3   00-02-15     
1.4   00-02-28     
1.5   00-03-13     
1.6   00-05-25  released for SFB 382 review
1.7   00-06-19  socket connection updated
1.8   00-07-06  windows port done
1.9   00-12-11  new transfer function definition
1.91  01-03-13  - new software rendering algorithm
                - doxygen-style source code documentation
1.92  01-03-16  - new program names: vshell, vview, vconv
                - updated directory structure
1.93  01-03-30  - more detailed error messages at program start
                - selectable interpolation mode: linear/nearest neighbor
                - new rendering algorithm (based on shear-warp factorization)
                - new viewer choice in VShell
1.94  01-04-23  - new volume file format: AVF
1.95  01-05-03  - boundary box now drawn correctly with all renderers
                - user defined background color
                - VF file format features run length encoding of the 
                  volume data (8 bit scalars only)
                - new VConv command line parameters: flip, rotate, dist, crop, 
                  bits, resize, scale, sphere, interpolation, swap, shift, 
                  bitshift, transfunc
                - new VView command line parameters: renderer, size, 
                  perspective, boundaries, stereo, orientation, fps, transfunc
1.96  01-10-18  - adjustable image quality in VView                  
                - new PPM image file loader
                - enhanced DICOM file loader
                - new alpha pin type: hat pin
                - discrete colors in transfer function
1.97  02-09-04  - VView now features auto-rotation when the mouse button 
                  is released while in motion
                - new remote renderer on the basis of the shear-warp algorithm
                  introduced: VRemote. Uses MPI for parallelization.
                - new socket library features timeouts and more reliable 
                  connections
                - TCP volume transfer now uses new data format: RVF and XVF
                  binary formats can be used instead of complicated previous format
                - VConv: croptime parameter added for extraction of time steps
                  in volume animation
                - VShell: Reload button added to Movie Script window
                - New movie script format which allows loops. 
                  Old movie scripts will not work anymore.
                - VShell now uses Java 1.3.1.
                - XVF files now support RLE encoding by default by incorporating 
                  routines from the previous VF format
                - VShell: slice viewer supports multiple slicing directions and 
                  optionally uses the transfer function for the images
1.98  03-06-24  - VConv: new -stat option for volume data statistics
                - VConv: new -fillrange option for optimized scalar data range
                - VConv: new -dicomrename option to rename DICOM files 
                  according to header information
                - VConv: new option -nocompress to save xvf files without
                  RLE compression (compatible to earlier format)
                - VConv: new option -time to set the time step duration
                - VConv: multiple slice files or time steps must now be
                  loaded with -files instead of -timesteps. Collections of slice 
                  files will no longer be read automatically.
                - VConv: new -deinterlace option to correct interlaced slices
                - VConv: new -pos option to set the position of the volume
                - VConv: new -hist option to print histogram information
                - VConv: new -zoom option to zoom data range
                - VConv: new -sign option to toggle sign of data values
                - VConv: new -blend option to blend two volumes together
                - VView: animation speed now adjustable
                - VView: replaced 'random peak' transfer function option with 
                  a peak which
                  can be shifted left and right in the transfer function domain
                - VShell: new screenshot feature in File menu
                - New reader for BrainVoyager VMR and VTC files
                - New reader for Gordon Kindlmann's Teem volume files
                - Read and write support for NRRD files (Gordon Kindlmann's 
                  Teem format)
                - Performance measurements under Windows now much more exact 
                  with QueryPerformance API
1.99b           - This is the first version released after I changed from 
                  the University of Stuttgart to Brown University
                - The XVF and NRRD file formats can now store multi-modal data
                - VConv: new -channels parameter to change the number of 
                  data channels
                - uncompressed 2D TIFF images can now be read
                - register combiners are now default algorithm for indexed mode
                - VShell: shows real data value in addition to data value in TFE
                - VShell: new menu item: File/Load Slices: creates volume 
                  from 2D slice images
                - VConv: -line renamed to -drawline
                - VConv: new command -drawbox to draw a solid 3D box in a volume
                - XVF files now support an icon
                - VConv: new parameters: -makeicon, -seticon, -geticon
                - Got rid of user interfaces for the shear-warp and remote 
                  renderers for the sake of a simpler API. Will put them back
                  in once they have an edge over GPU based approaches.
                - VView: new parameter -voxeltype to set the rendering algorithm
                - VView: new right-button menu: voxel representation
1.991b          - Now supports multiple channels of types: 8 bit, 16 bit, float
                - Optionally uses pixel shader for RGB volumes
                  (requires GeForce FX)
                  for real-time transfer function editing
                - VConv: added -swapChannels command
                - Renamed 'modality' to 'channel' for clarity's sake.
                - VConv: renamed -zoom to -zoomdata and added channel parameter
                - VConv: renamed -bits to -bpc
                - VConv: new command -makevolume to compute volume data                
2.00b           - Major version change to 2.0: GUI is now based no FOX Toolkit,
                  rather than Java. This gives much higher reliability and easier
                  installation.
                - New support for floating point data sets (float data type)
                - GUI support to change range of floating point data to be rendered.
                - DeskVOX: several volume manipulation routines (crop, resize, etc)
2.01b           - VConv: added -heightfield parameter
                - DeskVOX: new dialog box for height fields     
                - First version that is open-source under the LGPL license.           
                - VConv: added -autodetectrealrange
                - DeskVOX: added region of interest (ROI) rendering mode        
                - DeskVOX: added red-green/red-blue stereo modes
                - TIFF files now support 16 bit pixels
                - Added writer for AVF (ASCII) volume files
                - Channel with gradient magnitude can now be generated in DeskVOX.
                - XVF file format now stores channel names.
                - XVF file format has new ASCII-header for easier debugging and 
                  more flexible adding of parameters.
                - Transfer functions completely revised: multi-dimentional TF widgets available
                - VConv: new -mergetype parameter
                - Deskvox: added auto rotation mode to View menu: object keeps spinning when mouse 
                  button is released in motion
                - Deskvox: added Skip Range widget in transfer function editor: 
                  forces a range to be transparent
                - Deskvox: new Import TF button in TFE to load TF from other file
                - Deskvox: made output image file specification easier in Screenshot and Moviescript dialogs
                - Deskvox: added new movie script commands for clipping plane control: 
                  setclip, moveclip, setclipparam
                - New Custom widget allows entering transfer functions with control points.
                - In Floating Point Range dialog box: new buttons for high-dynamic range transfer functions.
                - Deskvox: saved Meshviewer style transfer functions use new Custom widget for opacity
                  
                
Distribution
------------

You may copy and distribute this program as you like,
provided that the distribution files remain unaltered and complete.
No one except the author is allowed to accept money for this software.
Using parts or all of the program in other projects, or modification of
the distribution is prohibited, unless explicitly allowed by the author. 
Any type of re-engineering of the compiled code is also prohibited. 
If you are interested in the source code, please contact the author for 
the latest licensing status.


Acknowledgements
----------------

I would like to thank the following people (in no particular order):

- Philippe Lacroute from Stanford University for implementing the shear-warp 
  algorithm and making his API publicly available as the Volpack library.
- Roland Niemeier for the implementation of clipping planes and a first version 
  of the perspective projection algorithm. I also thank him for his 
  extensive effort in explaining his algorithms and source code to me.
- Daniel Weiskopf for the spherical textures renderer and also for helping
  me understand volume rendering and OpenGL issues.
- Sven Wergandt for writing the volume file format routines. His RLE encoding
  is very useful to save storage space.
- Uwe Woessner for, among countless other things, adapting the COVISE system 
  to make the Virvo VR plugin work. He and I also developed a virtual reality 
  transfer function editor which is described in our IPTW/EGVE 2001 conference 
  paper.
- Ulrich Lang, my advisor in Stuttgart, who always pointed me in the right directions 
  and helped me keep the big picture when I got lost in detail.
- Marc Schreier for providing excellent user feedback on the Virvo distribution
  and for marvellous volume data sets, as well as for the TGA file loader.
- Guenter Knittel for pointing me to the QueryPerformance API
- Alexander Rice for the pixel shader integration, as well as porting the VShell
  Java GUI to the FOX Toolkit.


Disclaimer
----------

DISCLAIMER OF WARRANTY:
THE PROGRAM AND/OR THE SOURCE CODE IS PROVIDED "AS IS." ALL EXPRESS AND IMPLIED 
WARRANTIES AND CONDITIONS ARE DISCLAIMED, INCLUDING, WITHOUT LIMITATION, ANY IMPLIED 
WARRANTIES AND CONDITIONS OF MERCHANTABILITY, SATISFACTORY QUALITY, FITNESS 
FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. SHOULD THE SOFTWARE PROVE 
DEFECTIVE IN ANY RESPECT, NOBODY ASSUMES ANY COST OR LIABILITY FOR SERVICING, 
REPAIR OR CORRECTION. THIS DISCLAIMER OF WARRANTY IS AN ESSENTIAL PART OF THIS 
LICENSE. NO USE OF ANY COVERED CODE IS AUTHORIZED HEREUNDER EXCEPT SUBJECT TO 
THIS DISCLAIMER.

LIMITATION OF LIABILITY:
UNDER NO CIRCUMSTANCES NOR LEGAL THEORY, WHETHER TORT (INCLUDING, WITHOUT 
LIMITATION, NEGLIGENCE OR STRICT LIABILITY), CONTRACT, OR OTHERWISE, WILL
ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR 
CONSEQUENTIAL DAMAGES OF ANY CHARACTER INCLUDING, WITHOUT LIMITATION, DAMAGES 
FOR LOSS OF GOODWILL, WORK STOPPAGE, LOSS OF DATA, COMPUTER FAILURE OR 
MALFUNCTION, OR ANY AND ALL OTHER COMMERCIAL DAMAGES OR LOSSES, EVEN IF SUCH 
PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

INDEMNITY:
THE USER OF THIS SOFTWARE SHALL BE SOLELY RESPONSIBLE FOR DAMAGES ARISING, 
DIRECTLY OR INDIRECTLY, OUT OF ITS UTILIZATION OF RIGHTS UNDER THIS LICENSE. 
THE USER WILL DEFEND, INDEMNIFY AND HOLD HARMLESS EVERYBODY FROM AND AGAINST 
ANY LOSS, LIABILITY, DAMAGES, COSTS OR EXPENSES (INCLUDING THE PAYMENT OF 
REASONABLE ATTORNEYS FEES) ARISING OUT OF THE USER'S USE, MODIFICATION, 
REPRODUCTION AND DISTRIBUTION OF THE COVERED CODE OR OUT OF ANY REPRESENTATION 
OR WARRANTY MADE BY THE USER.

SOURCE CODE:
THE ORIGINAL SOURCE CODE WAS DEVELOPED BY JURGEN P. SCHULZE AT THE 
UNIVERSITY OF STUTTGART (GERMANY) AND BROWN UNIVERSITY (PROVIDENCE, RI, USA). 
THE SOURCE CODE IS COPYRIGHT (C) 1999-2005 BY JURGEN P. SCHULZE. 
COPYRIGHT IN ANY PORTIONS CREATED BY THIRD PARTIES 
IS AS INDICATED ELSEWHERE HEREIN. ALL RIGHTS RESERVED.
