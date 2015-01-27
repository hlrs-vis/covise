#ifndef CO_WIIMOTE_H
#define CO_WIIMOTE_H

#include <cstdlib>
#include <util/coExport.h>
#include <device/ButtonDevice.h>

#ifdef HAVE_WIIMOTE
extern "C" {
#include <wiimote.h>
}
#else
struct wiimote_t;
#endif

class Wiimote : public opencover::ButtonDevice
{
private:
    bool update();
    bool tryToConnect();

    wiimote_t *wiimote;
    bool connected;
    int wheelcounter;
    std::string WiiAddress;

public:
    Wiimote();
    ~Wiimote();
    void getButtons(int station, unsigned int *status);
    int getWheel(int station);
    void reset();
};
#endif
