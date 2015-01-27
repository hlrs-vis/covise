         TrackIRServer
         Virtual Reality Center Hochschule Mannheim - University of Applied Sciences, 2010
         P.Gehrt@HS-Mannheim.de

 - This plugin receives data of tracked objects from TrackIRServer and passes the provided data as a matrix to covise.
   It can be used for headtracking or handtracking. Currently, only one object can be tracked at a time.

 - Requires the "TrackIRServer"-module running.

 - Adjust the ip-address and ports at both sides, default value is localhost:7050. Firewall adjustments may be necessary.

 - This plugin has to be activated at Covise startup, modify the <Input> section of your config.xml as follows:

<TrackingSystem value="TrackIRPlugin">

   <Offset x="0" y="0" z="0" />

   <Orientation h="0" p="0" r="0" />

</TrackingSystem>


