# coVRAudio 

This directory contains the source for the library target `coVRAudio`, which
implements audio features for OpenCOVER.

It supports multiple backends (via `Player` subclasses). 

The library is built unconditionally, so you can always link against it in your
plugin code if you need audio support. For reading audio files, OpenAL and ALUT
are required, and if OpenCOVER is compiled without these libraries, the
`coVRAudio` library will still be present, but a player will not be initalized,
and a message is printed to the command line on initialization for the user to
understand that audio support is unavailable.

## Usage

The namespace of all classes in this library is `opencover::audio`. 

The `Player` class is the abstraction of the different backends, which is
instantiated by `coVRPluginSupport` (`cover->getPlayer()`). All players allow
creation of a corresponding `Source`, which can be positioned in 3D space and
played/stopped. It requires an audio file to be loaded via the `Audio` class.
A common pattern is the following:

```cpp
#include <audio/Player.h>
#include <audio/Audio.h>
#include <audio/Source.h>

using namespace opencover::audio;

class MyPlugin {
public:
    void init() {
        auto player = cover->getPlayer();

        // Check if audio support is enabled and configured
        if (player) {
            // Load the audio file
            m_audio.setURL("path/to/file.wav");

            // Initialize the source for playback with the active player
            m_source = player->makeSource(&m_audio);
        }
    }

    void someEvent() {
        if (m_source) {
            // Position the audio and play it
            m_source->setPosition(10.f, 20.f, 30.f);
            m_source->setIntensity(0.5f);
            m_source->play();
        }
    }

private:
    Audio m_audio;
    std::shared_ptr<Source> m_source;
};
```
