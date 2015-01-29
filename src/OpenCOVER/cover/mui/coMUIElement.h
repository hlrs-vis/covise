#ifndef CO_MUI_ELEMENT_H
#define CO_MUI_ELEMENT_H

/*! file
 * \brief user interface proxy class
 *
 * \author Andreas Grimm
 * \date
 */

#include <OpenVRUI/sginterface/vruiMatrix.h>
#include <util/coTypes.h>
#include <util/coDLList.h>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <queue>
#include <map>
#include <tui/coAbstractTabletUI.h>
#include <QObject>
#include <string>
#include <OpenVRUI/coUIContainer.h>

namespace coMUI
{
class coMUIElement;
}

namespace osg
{
class Node;
}

namespace opencover
{
class coTabletUI;
class coTUIElement;
class TextureThread;
class SGTectureThread;
class LocalData;
class IData;
class IRemoteData;
class coTUIListener;
}

namespace covise
{
class coUIContainer;
class coUIUserData;
class TokenBuffer;
class Hose;
class Message;
class VRBClient;
class Connection;
class ClientConnection;
class ServerConnection;
class coUIUserData;
class vruiTransformNode;
class vruiUIElementProvider;
}

class coMUIConfigManager;

/*
 *Base class for MUI elements
 */
class COVEREXPORT coMUIElement: public QObject
{

    Q_OBJECT

public:
    // constructor:
    // n: shown Name
    // pID:  Parent ID
    coMUIElement();
    coMUIElement(const std::string &n, int pID);
    coMUIElement(const std::string &n, const std:: string &l, int pID);

    struct device{
        std::string UI;
        std::string Device;
        std::string Identifier;
        bool Visible;
        std::string Label;
    };

    // destructor:
    ~coMUIElement();

    // methods:
    virtual bool fileExist(std::string File);
    std::string findLabel(const std::string Instanz, std::string label, std::string keywordUI, std::string keywordDevice);

    // must be overwritten, if inherited:
    virtual void setPos(int posx, int posy);            // must be overwritten, if inherited
    virtual std::string getUniqueIdentifier();          // must be overwritten, if inherited

private:
    coMUIConfigManager *ConfigManager;
protected:
    int ID;                                     // ID of the elements
    std::string name_str;                       // name of the elements
    std::string label_str;                      // label of the elements

};


#endif
