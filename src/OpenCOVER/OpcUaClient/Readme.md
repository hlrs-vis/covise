Opc-Ua client based on open62541 library
-----------------------------------------
Connect an Opc-Ua client via connect(name) or use an already connected client via getClient(name). The name refers to a config entry which contains the connection details. The can also be edited in the tabletUI.

After connect() the client tries to establish an asynchronous connection to then wait for changes of observed variables int this extra thread. Users can observe nodes via the clients observeNode(nodeName) method until the ObserverHandle returned by this method is deleted.
The client provides getters to retrieve the updates in the main thread. The will return the first received update that has not been gotten yet.
Warning: if a client is used by multiple instances this will cause problems!
