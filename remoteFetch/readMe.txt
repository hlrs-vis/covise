How to use remote fetch:

 1: config:
	set in 'config/system/vrb/RemoteFetch value = "on" to enable remote fetch in local usr tmp directory. 

	/set in 'config/system/vrb/RemoteFetch path="your path" to chose a differen directory to remote Fetch to. 
example to remote fetch in this directory:
  <SYSTEM>
   <VRB>
    <Server value="visent.hlrs.de" tcpPort="31251" udpPort="31253" /> //use your connection specifications
	<RemoteFetch value="on" path="%COVISE_PATH%/remoteFetch" />
   </VRB>
  </SYSTEM>
  
  2. Start OpenCOVER
  3. join a session with the partner that has the file 
	-> if the file was loaded remote fetch will begin
  4. partner loads the file
  
Notes:
dependent files that get loaded by the main file currently are only transmittet for vrml files