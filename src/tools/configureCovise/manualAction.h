/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

class manualAction
{
public:
    void hostConfigItem(
        const char *nameOfHost,
        int sharedMemory, const char *strSharedMemory,
        int executionMode, const char *strExecutionMode,
        int timeout, const char *hostname);
    void pipeConfigItem(
        char *hostname,
        int softPipe,
        int hardPipe,
        const char *display);
    void windowConfigItem(
        int winNo,
        const char *name,
        int softPipeNo,
        int origX,
        int origY,
        int sizeX,
        int sizey,
        const char *hostname);
    void channelConfigItem(
        int channNo,
        const char *channelName,
        int winNo,
        double viewPortXMin,
        double viewPortYMin,
        double viewPortXMax,
        double viewPortYMax,
        const char *hostname);
    void screenConfigItem(
        int screenNo,
        const char *screenName,
        int hSize,
        int vSize,
        int origX,
        int origY,
        int origZ,
        double head,
        double pitch,
        double roll,
        const char *hostname);
    void vrviewpointsItem(
        const char *viewpointName,
        float scale,
        float x,
        float y,
        float z,
        float h,
        float p,
        float r,
        const char *hostname);
    void licenseItem(
        const char *key,
        const char *name,
        const char *date,
        const char *hostname);
    void buttonConfigMAP(char *hostname, int number, int buttonTypeNumber, const char *button);
    void createStringList(const char *string);
    void addToStringList(const char *string);
    void CoverWelcome(const char *hostname);
    void UIShortCuts(const char *hostname);
    void colormapsitem(const char *name);
    void colormapRGB(const char *hostname, const char *colorMapName, int x, int y, int z);
    void colormapRGBA(const char *hostname, const char *colorMapName, double x, double y, double z, double w);
    void colormapRGBAX(const char *hostname, const char *colorMapName, double x, double y, double z, double v, double w);
    void COVERModule(const char *hostname, const char *modules);
};
