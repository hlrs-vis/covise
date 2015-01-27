Copyright (C) 2007-2013, Advanced Realtime Tracking GmbH



License
-------

This library is distributed under the GNU LGPL (GNU Lesser General Public License). 
You can modify and/or include the sources into own software (for details see 
'license_lgpl.txt').



Purpose of DTrackSDK
--------------------

A set of functions to provide an interface to A.R.T. tracking systems. It supports
both DTrack and DTrack2. 
The functions receive and process DTrack/DTrack2 measurement data packets (UDP; ASCII),
and send/exchange DTrack/DTrack2 command strings (UDP/TCP; ASCII).



How to receive and process A.R.T. tracking data
-----------------------------------------------

DTrack/DTrack2 uses ethernet (UDP/IP datagrams) to send measurement data to other
applications. They both use an ASCII data format.
In its most simple operating mode DTrackSDK is just receiving and processing these data. In
this case DTrackSDK just needs to know the port number where the data are arriving; all necessary
settings have to be done manually in the DTrack/DTrack2 frontend software.

DTrack/DTrack2 also provides a way to control the tracking system through a command interface via
ethernet. Both DTrack and DTrack2 use ASCII command strings. DTrack2 commands are sent via a
TCP/IP connection.

The formats and all other necessary definitions are described in
'ARTtrack & DTrack(2) Manual: Technical Appendix'.



Sample source codes for an own interface
----------------------------------------

The sample source code files show how to use the DTrackSDK:

	example_without_remote_control.cpp:       sample without usage of remote commands (C++)
	example_with_simple_remote_control.cpp:   simple dtrack/dtrack2 sample with usage of remote commands (C++)
	example_with_dtrack2_remote_control.cpp:  dtrack2 sample with usage of remote commands (C++)
	example_listen_to_multicast.cpp:          multicast sample without usage of remote commands (C++)
	example_fingertracking.cpp                fingertracking sample

Each example uses a different constructor and explains how to use it.

All examples are written in C++, and work for both Unix and Windows. The files have been
successfully tested under Linux, Windows 2000 and Windows XP.

For Windows: please link with library 'ws2_32.lib'.



Compatibility with older SDK versions
-------------------------------------

DTrackSDK comes with additional sources that should make it easier to upgrade from older SDK
versions. Actually they are just wrappers around class 'DTrackSDK', and allow to use the known
interface. They are available for the classes:

	DTracklib:  class only for DTrack systems (C++; used from May 2005 until Dec 2006)
	DTrack:     class only for DTrack systems (C++; used from Jan 2007 until Dec 2010)
	DTrack2:    class only for DTrack2 systems (C++; used from May 2008 until Dec 2010)

Note that the wrappers just work for the C++ SDK versions; there's no wrapper for the 'dtracklib'
SDK written in C.



Source Documentation
--------------------

Please refer to ./doc/html/ 



Contents of this package
------------------------
 
- /:
 	DTrackSDK.hpp,
	DTrackSDK.cpp:  C++ class for receiving and processing DTrack/DTrack2 tracking data (ASCII),
	                and for sending/exchanging commands to DTrack/DTrack2
	example_without_remote_control.cpp:	 sample without usage of remote commands (C++)
	                                     uses 'DTrackSDK(dataport)' constructor
	example_with_simple_remote_control.cpp:   sample with usage of remote commands (C++)
	                                          suitable for DTrack/DTrack2
	                                          uses 'DTrackSDK(server_host, server_port, data_port)' constructor
	example_with_dtrack2_remote_control.cpp:  sample with usage of remote commands (C++)
	                                          uses 'DTrackSDK(server_host, data_port)' constructor
	example_listen_to_multicast.cpp:     multicast sample without usage of remote commands (C++)
	                                     explains how to enable multicast in DTrackSDK
 
- /Lib:
	DTrackDataType.h:  type definitions	
	DTrackNet.h,
	DTrackNet.cpp:     functions for sending / receiving data
	DTrackParse.hpp,
	DTrackParse.cpp:   functions for processing data
	DTrackParser.hpp,
	DTrackParser.cpp:  parser class to process data

- /Compatibility/DTrackLib:
	DTracklib.hpp,
	DTracklib.cpp:  older C++ class for receiving and processing DTrack tracking data (ASCII),
	                and for sending commands to DTrack
	example_without_remote_control.cpp:	 sample without usage of remote commands (C++)
	example_with_remote_control.cpp:     sample with usage of remote commands (C++)

- /Compatibility/DTrack:
	DTrack.hpp,
	DTrack.cpp:  older C++ class for receiving and processing DTrack tracking data (ASCII),
	             and for sending commands to DTrack
	example_without_remote_control.cpp:  sample without usage of remote commands (C++)
	example_with_remote_control.cpp:     sample with usage of remote commands (C++)

- /Compatibility/DTrack2:
	DTrack2.hpp,
	DTrack2.cpp:  older C++ class for receiving and processing DTrack2 tracking data (ASCII),
	              and for exchanging command strings with a DTrack2 server
	example_without_remote_control.cpp:  sample without usage of remote commands (C++)
	example_with_remote_control.cpp:     sample with usage of remote commands (C++)



Development with DTrackSDK
--------------------------

a) Create a new project using the DTrackSDK:

	- include the header file:
		DTrackSDK.hpp
	- add following include paths:
		./
		./Lib
	- add following source files to your project:
		./DTrackSDK.cpp
		./Lib/DTrackNet.cpp
		./Lib/DTrackParse.cpp
		./Lib/DTrackParser.cpp
	- you may want to start with one of the example files provided in this package

b) Upgrade an existing project developed with older DTrack SDK versions:

	- upgrade from class 'DTracklib':
		~ include the header file:
			DTracklib.hpp
		~ add following include paths
			./
			./Lib
			./Compatibility/DTrackLib/
		~ add following source files to your project:
			./DTrackSDK.cpp
			./Lib/DTrackNet.cpp
			./Lib/DTrackParse.cpp
			./Lib/DTrackParser.cpp
			./Compatibility/DTrackLib/DTracklib.cpp

	- upgrade from class 'DTrack':
		~ include the header file:
			DTrack.hpp
		~ add following include paths
			./
			./Lib
			./Compatibility/DTrack/
		~ add following source files to your project:
			./DTrackSDK.cpp
			./Lib/DTrackNet.cpp
			./Lib/DTrackParse.cpp
			./Lib/DTrackParser.cpp
			./Compatibility/DTrack/DTrack.cpp
			
	- upgrade from class 'DTrack2':
		~ include the header file:
			DTrack2.hpp
		~ add following include paths
			./
			./Lib
			./Compatibility/DTrack2/
		~ add following source files to your project:
			./DTrackSDK.cpp
			./Lib/DTrackNet.cpp
			./Lib/DTrackParse.cpp
			./Lib/DTrackParser.cpp
			./Compatibility/DTrack2/DTrack2.cpp



Company details
---------------

Advanced Realtime Tracking GmbH
Am Oeferl 6
D-82362 Weilheim
Germany

http://www.ar-tracking.de/
