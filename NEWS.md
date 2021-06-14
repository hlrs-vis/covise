
Recent changes
====================

This document only covers changes that are important for using Covise and OpenCOVER. This includes changes on the UI, the config and requirements.
For technical changes please have a look at the git commits.

14.06. 2021 Adjusted config entry for COVER plugins
---------------------------------------------------
Added new attribute to COVER.Plugin entries: shared="on"/"off".
If set on the plugin will be loaded in a collaborative session if a partner load the plugin. 


03.05.2021 Integrated OpenCOVER's remote daemon in the coviseDaemon
------------------------------------------------------------------
This daemon is used to start COVER slaves in a multi display environment. This feature can be enabled for the coviseDaemon by adding the config entry <Daemon port="your port" /> to the COVER section.
For OpenCOVER to use this, also in the COVER section, set NumSlaves to >= 1 and add a line like
    <Startup value="startOpenCover" host="your host" name="your slave id" /> 
for every slave.

26.04.2021 Renamed VrbRemoteLauncher to coviseDaemon
----------------------------------------------------

14.04.2021 Added new config entry: System.VRB.CheckConnectionTimeout
--------------------------------------------------------------------
This specifies the time in seconds that COVISE waits for a direct connection to a new partner.
If the direct connection times out a proxy connections is established via VRB.

13.04.2021 COVISE-proxy integrated in VRB
-----------------------------------------
If the IP-address of the COVISE-host is not routable partners will be added by routing all traffic through the VRB


19.03.2021 Start of collaborative sessions
------------------------------------------
In OpenCOVER potential partners (that have a VrbRemoteLauncher running) are shown under File/Launch remote COVER and can be selected to add a OpenCOVER partner to the current session.
In COVISE the CSCW menu has been reworked and now also shows a list of running VrbRemoteLauncher via which partners or hosts can be added. (Note: the only old start options remaining are manual and script)


09.12.2020 VrbRemoteLauncher
----------------------------
Moved command line argument options from main window to side menu and added hotkeys: Enter to connect, ESC to cancel or disconnect, CTRL + Tab to toggle menu.

30.11.2020 Launch partner OpenCOVER directly from file menu
-----------------------------------------------------------
Added a menu entry (Launch remote COVER) in the file menu that shows available VrbRemoteLauncher. Click on a partner to let him join you session.


17.11.2020 VrbRemoteLauncher 
----------------------------
App that launches and request launches of COVISE and OpenCOVER via a VRB server.

12.09.2019 Changes to VRB connection menu in OpenCOVER
-------

The buttons for creating and joining VRB-sessions moved from the extra "connections" menu item to the "file" menu. Renamed "available sessions" to "current session".



