Make OpenCOVER load the plugin automatically during startup,
e.g. make sure that $COVISEDIR/config/config.xml contains something
similar to the following:

<?xml version="1.0"?>

<COCONFIG version="1" >
  <GLOBAL>
    <COVER>
      <Plugin>
        <!-- the following line enables the FileLoader plugin during OpenCOVER startup -->
        <FileLoader value="on" />
      </Plugin>
    </COVER>
  </GLOBAL>
</COCONFIG>


For more information have a look at the comments in the source code.
