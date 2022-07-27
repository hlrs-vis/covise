Stream plugin
-------------

This plugin streams the COVER image to a virtual webcam created with akvcam.
For this to work akvcam must be installed setup and loaded.
First install akvcam:

    git clone https://github.com/webcamoid/akvcam.git
    cd akvcam/src
    make 
    sudo make dkms_install

To configure akvcam place example configuration file under

    /etc/akvcam/config.ini

This modules default settings match this default config.

Now load the module:

    sudo modprobe akvcam 

/dev/videxX and /dev/videoY should be created, where X and Y are numbers given by the operating system.
Now streaming can begin by x as the video device number and selecting the virtual camera in your conferencing tool.

Optional: place a fallback image under

    /etc/akvcam/default_frame.bmp 

to prevent flimmers while COVER stream is turned of.
