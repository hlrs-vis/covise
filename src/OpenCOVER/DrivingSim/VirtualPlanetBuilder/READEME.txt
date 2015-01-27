VirtualPlanetBuilder is terrain database creation tool that is able to read 
a wide range of geospatial imagery and elevation data and build from small 
area terrain database to massive whole planet paged databases. These databases 
can then be uploaded onto the internet and provide online GoogleEarth style 
roaming of whole planet databases, or kept on local disks for high speed 
access such as required for professional flight simulators.

VirtualPlanetBuilder just builds databases so for runtime viewing of database 
you'll need an OpenSceneGraph based application. The VirtualPlanetBuilder 
itself is based on the OpenSceneGraph real-time graphics toolkit, and creates 
databases in native OpenSceneGraph binary format for maximum paging performance. 
For non OpenSceneGraph based applications you'll need to convert the database
by writing your own post processing tool to convert from OpenSceneGraph native
format into you own native formats, or convert to COLLADA using the 
OpenSceneGraph native support for reading and writing COLLADA.

The VirtualPlanetBuilder project grew from the original paged database 
generation tools that can as part of the !OpenSceneGraph-1.2, and is now a 
separate project to enable both projects to focus on their own core disciplines.
We have plans to make it even more scalable making it possible to create 
multi-terabyte paged databases, and to create them across networks of 
computers, each of which take part of the job of creating the complete 
database. We will also provide support for database optimization for the 
lower bandwidth constraints of web based 3d database visualization.


Robert Osfield.
22nd June 2008
