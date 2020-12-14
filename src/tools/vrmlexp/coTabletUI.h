/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TABLET_UI_H
#define CO_TABLET_UI_H

 /*! \file
  \brief Tablet user interface proxy classes

  \author Uwe Woessner <woessner@hlrs.de>
  \author (C) 2004
		  High Performance Computing Center Stuttgart,
		  Allmandring 30,
		  D-70550 Stuttgart,
		  Germany

  \date
  */

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#include <time.h>
#endif
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <queue>
#include <map>
#include <list>
#include <string>

#ifndef _M_CEE //no future in Managed OpenCOVER
#include <future>

#endif
  //#ifndef WIN32
  //#include <stdint.h>
  //#define FILESYS_SEP "\\"
  //#else
#define FILESYS_SEP "/"
//#endif
class coAbstractTabletUI;
class coAbstractTUIElement;
class ClientConnection;
class TokenBuffer;
class Message;
class coTUITabFolder;

/// Action listener for events triggered by any coAbstractTUIElement.
class coAbstractTUIListener
{
public:
	/** Action listener for events triggered by coAbstractTUIElement.
	@param tUIItem pointer to element item which triggered this event
	*/
	virtual ~coAbstractTUIListener()
	{
	}
#if 0
	virtual void tabletEvent(coAbstractTUIElement* tUIItem) = 0;
	virtual void tabletPressEvent(coAbstractTUIElement* tUIItem) = 0;
	virtual void tabletSelectEvent(coAbstractTUIElement* tUIItem) = 0;
	virtual void tabletFindEvent(coAbstractTUIElement* tUIItem) = 0;
	virtual void tabletReleaseEvent(coAbstractTUIElement* tUIItem) = 0;
	virtual void tabletCurrentEvent(coAbstractTUIElement* tUIItem) = 0;
#endif
};

/**
* Tablet PC Userinterface Mamager.
* This class provides a connection to a Tablet PC and handles all coAbstractTUIElement.
*/
class coAbstractTabletUI
{
public:
	virtual ~coAbstractTabletUI()
	{
	}
	virtual bool update() = 0;
};

/**
* Base class for Tablet PC UI Elements.
*/
class coAbstractTUIElement
{
public:
	virtual ~coAbstractTUIElement()
	{
	}
	virtual void parseMessage(TokenBuffer& tb) = 0;
	virtual void resend(bool create) = 0;
	virtual void setPos(int, int) = 0;
	virtual void setSize(int, int) = 0;
	virtual void setLabel(const char* l) = 0;
	virtual coAbstractTUIListener* getMenuListener() = 0;
};

/**
* the filebrowser push button.
*/
class coAbstractTUIFileBrowserButton : public coAbstractTUIElement
{
public:
	enum DialogMode
	{
		OPEN = 1,
		SAVE = 2
	};
	virtual ~coAbstractTUIFileBrowserButton()
	{
	}
	virtual void setDirList(Message& ms) = 0;
	virtual void setFileList(Message& ms) = 0;
	virtual void setCurDir(Message& msg) = 0;
	virtual void setCurDir(const char* dir) = 0;
	virtual void resend(bool create) = 0;
	virtual void parseMessage(TokenBuffer& tb) = 0;
	virtual void setDrives(Message& ms) = 0;
	virtual void setClientList(Message& msg) = 0;
};
class coTabletUI;
class coTUIElement;
class SGTextureThread;
class LocalData;
class IData;
class IRemoteData;
#ifdef WIN32
#pragma warning(push)
#pragma warning(disable: 4275)
#endif
/// Action listener for events triggered by any coTUIElement.
class coTUIListener : public coAbstractTUIListener
{

public:
	/** Action listener for events triggered by coTUIElement.
	  @param tUIItem pointer to element item which triggered this event
	  */
	virtual ~coTUIListener()
	{
	}
	virtual void tabletEvent(coTUIElement* tUIItem);
	virtual void tabletPressEvent(coTUIElement* tUIItem);
	virtual void tabletSelectEvent(coTUIElement* tUIItem);
	virtual void tabletChangeModeEvent(coTUIElement* tUIItem);
	virtual void tabletFindEvent(coTUIElement* tUIItem);
	virtual void tabletLoadFilesEvent(char* nodeName);
	virtual void tabletReleaseEvent(coTUIElement* tUIItem);
	virtual void tabletCurrentEvent(coTUIElement* tUIItem);
	virtual void tabletDataEvent(coTUIElement* tUIItem, TokenBuffer& tb);
};

#ifdef WIN32
#pragma warning(pop)
#endif



#define THREAD_NOTHING_TO_DO 0

	class TokenBuffer;
	class Host;
	class Message;
	class Connection;
	class ClientConnection;
	class ServerConnection;
namespace osg
{
	class Node;
};
	class coTabletUI;
	class coTUIElement;
	class SGTextureThread;
	class LocalData;
	class IData;
	class IRemoteData;

	/**
	 * Tablet PC Userinterface Mamager.
	 * This class provides a connection to a Tablet PC and handles all coTUIElements.
	 */
	class coTabletUI :public coAbstractTabletUI
	{


	private:
		static coTabletUI* tUI;
		OpenThreads::Mutex connectionMutex;


	public:
		coTabletUI();
		coTabletUI(const std::string& host, int port);
		virtual ~coTabletUI();
		static coTabletUI* instance();
		static void destroy() { delete tUI; };
		coTUITabFolder *mainFolder;

		int getID();
		static void CALLBACK coTabletUI::timerCallback(HWND hwnd, UINT uMsg, UINT_PTR idEvent, DWORD dwTime);


		virtual bool update();
		void addElement(coTUIElement*);
		void removeElement(coTUIElement* e);
		void send(TokenBuffer& tb);
		void tryConnect();
		void close();
		bool debugTUI();
		bool isConnected() const;

		void lock()
		{
			connectionMutex.lock();
		}
		void unlock()
		{
			connectionMutex.unlock();
		}
		Host* connectedHost = nullptr;

		bool serverMode = false;
		Connection* sgConn = nullptr;

	protected:
		void init();
		void resendAll();
		std::vector<coTUIElement*> elements;
		std::vector<coTUIElement*> newElements;
		ServerConnection* serverConn = nullptr;
		Host* serverHost = nullptr;
		Host* localHost = nullptr;
		int port = 31804;
		int ID = 3;
		float timeout = 1.f;
		bool debugTUIState = false;
		double oldTime = 0.0;
		bool firstConnection = true;

		Connection* conn = nullptr;
#ifndef _M_CEE //no future in Managed OpenCOVER
		std::future<Host*> connFuture;
#endif
	};


	/**
	 * Base class for Tablet PC UI Elements.
	 */
	class coTUIElement : public coAbstractTUIElement
	{

	public:
		coTUIElement(const std::string&, int pID = 1);
		virtual ~coTUIElement();
		virtual void parseMessage(TokenBuffer& tb) override;
		virtual void resend(bool create) override;
		virtual void setEventListener(coTUIListener*);
		virtual coTUIListener* getMenuListener() override;
		void createSimple(int type);
		coTabletUI* tui() const;

	public:
		void setVal(const std::string& value);
		void setVal(bool value);
		void setVal(int value);
		void setVal(float value);
		void setVal(int type, int value);
		void setVal(int type, float value);
		void setVal(int type, int value, const std::string& nodePath);
		void setVal(int type, const std::string& nodePath, const std::string& simPath, const std::string& simName);
		void setVal(int type, int value, const std::string& nodePath, const std::string& simPath);

		int getID() const;

		virtual void setPos(int, int) override;
		virtual void setSize(int, int) override;
		virtual void setLabel(const char* l) override;
		virtual void setLabel(const std::string& l);
		//virtual void setColor(Qt::GlobalColor);
		virtual void setHidden(bool);
		virtual void setEnabled(bool);
		std::string getName() const
		{
			return name;
		}

	protected:
		coTUIElement(const std::string&, int pID, int type);
		coTUIElement(coTabletUI* tui, const std::string&, int pID, int type);

		int type = -1;
		int parentID;
		std::string name; ///< name of this element
		std::string label; ///< label of this element
		int ID; ///< unique ID
		int xs, ys, xp, yp;
		//Qt::GlobalColor color;
		bool hidden = false;
		bool enabled = true;
		coTUIListener* listener = nullptr; ///< event listener
		coTabletUI* m_tui = nullptr;
	};

	/**
	 * a static textField.
	 */
	class  coTUILabel : public coTUIElement
	{

		

	private:
	public:
		coTUILabel(const std::string&, int pID = 1);
		coTUILabel(coTabletUI* tui, const std::string&, int pID = 1);
		virtual ~coTUILabel();
		virtual void resend(bool create) override;

	protected:
	};
	/**
	 * a push button with a Bitmap
	 */
	class  coTUIBitmapButton : public coTUIElement
	{

		

	private:
	public:
		coTUIBitmapButton(const std::string&, int pID = 1);
		coTUIBitmapButton(coTabletUI* tui, const std::string&, int pID = 1);
		virtual ~coTUIBitmapButton();
		virtual void parseMessage(TokenBuffer& tb) override;

		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();

	protected:
	};
	/**
	 * a push button.
	 */
	class  coTUIButton : public coTUIElement
	{

		

	private:
	public:
		coTUIButton(const std::string&, int pID = 1);
		coTUIButton(coTabletUI* tui, const std::string&, int pID = 1);
		virtual ~coTUIButton();
		virtual void parseMessage(TokenBuffer& tb) override;

		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();

	protected:
	};

	/**
	 * the filebrowser push button.
	 */
	class  coTUIFileBrowserButton : public coTUIElement
	{
	public:
		enum DialogMode
		{
			OPEN = 1,
			SAVE = 2
		};
		coTUIFileBrowserButton(const char*, int pID = 1);
		coTUIFileBrowserButton(coTabletUI* tui, const char*, int pID = 1);
		virtual ~coTUIFileBrowserButton();

		// Sends a directory list to TUI
		virtual void setDirList(Message& ms);

		// Sends a file list to TUI
		virtual void setFileList(Message& ms);

		// Sends the currently used directory to TUI
		// Uses setCurDir(char*)
		void setCurDir(Message& msg);

		// Sends the currently used directory to TUI
		void setCurDir(const char* dir);

		// Resends all FileBrowser required data to TUI
		virtual void resend(bool create) override;

		// Parses all messages arriving from TUI
		virtual void parseMessage(TokenBuffer& tb) override;

		// sends the list of VRB clients in a session to TUI
		void setClientList(Message& msg);

		// retieve currently used data object
		// either LocalData, VRBData or AGData
		IData* getData(std::string protocol = "");

		// Retrieves the instance of the VRBData of the FileBrowser
		IData* getVRBData();

		//Sends a list of drives to the TUI
		void setDrives(Message& ms);

		// Returns the filename to a file in the local file system
		// based on a URI-Filelocation e.g. vrb://visper.hlrs.de//mnt/raid/tmp/test.wrl
		std::string getFilename(const std::string url);

		// Returns a file handle based on a URI in above mentioned format.
		// Not yet implemented.
		void* getFileHandle(bool sync = false);

		// Sets the file browser dialog mode (SAVE or OPEN) --> DialogMode
		void setMode(DialogMode mode);

		// Sends the currently used filter list for the filebrowser to the TUI
		void setFilterList(std::string filterList);

		// Returns a string containing a selected path
		// What was that again?
		std::string getSelectedPath();

	protected:
		void sendList(TokenBuffer& tb);
		// Stores the list of files as retrieved
		// from the storage location, however it is
		// recreated upon request basis.
		std::vector<std::string> mFileList;

		// Stores the list of directories as retrieved
		// from the storage location, however it is
		// recreated upon request basis.
		std::vector<std::string> mDirList;

		// Stores a list fo clients. Is this actually used?
		std::vector<std::string> mClientList;

		// General data object for access to non-local storage locations
		// e.g. VRB, AccessGrid
		IRemoteData* mDataObj;

		// Data Object for access to local file system
		LocalData* mLocalData;

		// Data Object for access to AccessGrid data store
		IRemoteData* mAGData;

		// Generic Data object interface only providing basic
		// functionality, capable of holding all data object types
		IData* mData;

		// Stores the location from where to determine file information
		// either as IP address or hostname
		std::string mLocation;

		// Stores the IP address of the system the local OpenCOVER runs on
		std::string mLocalIP;

		// Stores the currently selected directory as selected in the filebrowser
		// dialog of the TUI
		std::string mCurDir;

		// ??
		std::string mFile;

		// Stores the dialog mode of this TUIFileBrowserButton instance
		// either Open dialog or save dialog, currently implementation is
		// only focused on file open.
		DialogMode mMode;

		// String that contains a list of file-extensions to be used when
		// creating file list for the file browser
		// e.g. "*.*;*.wrl"
		std::string mFilterList;

		// Id of the VRB client which is required to create some outgoing messages
		// to the VRB server.
		int mVRBCId;

		// Stores data objects related to their protocol identifier
		std::map<std::string, IData*> mDataRepo;
		typedef std::pair<std::string, IData*> Data_Pair;

		// Id of the TUIFileBrowserButton
		int mId;

		/**
		   * Member containing the selected save path when in SAVE mode
		   */
		std::string mSavePath;
	};

	class  coTUIColorTriangle : public coTUIElement
	{

		


	private:
	public:
		coTUIColorTriangle(const std::string&, int pID = 1);
		virtual ~coTUIColorTriangle();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public:
		virtual float getRed() const
		{
			return red;
		}
		virtual float getGreen() const
		{
			return green;
		}
		virtual float getBlue() const
		{
			return blue;
		}
		virtual void setRed(float red)
		{
			this->red = red;
		}
		virtual void setGreen(float green)
		{
			this->green = green;
		}
		virtual void setBlue(float blue)
		{
			this->blue = blue;
		}
		virtual void setColor(float r, float g, float b);
		//virtual void switchLocation(LocationType type);

		void tabletEvent();
		void tabletReleaseEvent();

	protected:
		float red;
		float green;
		float blue;
	};
	class  coTUIColorButton : public coTUIElement
	{
		


	private:
	public:
		coTUIColorButton(const std::string&, int pID = 1);
		virtual ~coTUIColorButton();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual float getRed() const
		{
			return red;
		}
		virtual float getGreen() const
		{
			return green;
		}
		virtual float getBlue() const
		{
			return blue;
		}
		virtual float getAlpha() const
		{
			return alpha;
		}
		virtual void setRed(float red)
		{
			this->red = red;
		}
		virtual void setGreen(float green)
		{
			this->green = green;
		}
		virtual void setBlue(float blue)
		{
			this->blue = blue;
		}
		virtual void setAlpha(float alpha)
		{
			this->alpha = alpha;
		}
		virtual void setColor(float r, float g, float b, float a);
		//virtual void switchLocation(LocationType type);

		void tabletEvent();
		void tabletReleaseEvent();

	protected:
		float red;
		float green;
		float blue;
		float alpha;
	};

	class  coTUIColorTab : public coTUIElement
	{
		


	private:
	public:
		coTUIColorTab(const std::string&, int pID = 1);
		virtual ~coTUIColorTab();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual float getRed() const
		{
			return red;
		}
		virtual float getGreen() const
		{
			return green;
		}
		virtual float getBlue() const
		{
			return blue;
		}
		virtual float getAlpha() const
		{
			return alpha;
		}
		virtual void setRed(float red)
		{
			this->red = red;
		}
		virtual void setGreen(float green)
		{
			this->green = green;
		}
		virtual void setBlue(float blue)
		{
			this->blue = blue;
		}
		virtual void setAlpha(float alpha)
		{
			this->alpha = alpha;
		}
		virtual void setColor(float r, float g, float b, float a);

	
		void tabletEvent();

	protected:
		float red;
		float green;
		float blue;
		float alpha;
	};

	class  coTUIFunctionEditorTab : public coTUIElement
	{
	public:
		static const int histogramBuckets = 256;
		int* histogramData;

		// my transfer function parameters: what is needed?
		// for 1D, only points.

		// They have the same values of virvo TF widgets!
		enum TFKind
		{
			TF_COLOR = 0,
			TF_PYRAMID = 1,
			TF_BELL = 2,
			//TF_SKIP = 3,
			TF_FREE = 4,
			TF_CUSTOM_2D = 5,
			TF_MAP = 6,
			TF_CUSTOM_2D_EXTRUDE = 11,
			TF_CUSTOM_2D_TENT = 12
		};

		struct colorPoint
		{
			float r;
			float g;
			float b;
			float x;
			float y;
		};

		struct alphaPoint
		{
			int kind;
			float alpha;
			float xPos;
			float xParam1;
			float xParam2;
			float yPos;
			float yParam1;
			float yParam2;
			int ownColor;
			float r;
			float g;
			float b;
			int additionalDataElems; //int additionalDataElemSize;
			float* additionalData;
		};

		std::vector<colorPoint> colorPoints;
		std::vector<alphaPoint> alphaPoints;

		int tfDim;

	public:
		coTUIFunctionEditorTab(const char* tabName, int pID = 1);
		virtual ~coTUIFunctionEditorTab();

		int getDimension() const;
		void setDimension(int);

		virtual void resend(bool create) override;
		void sendHistogramData();
		virtual void parseMessage(TokenBuffer& tb) override;
	};


	/**
	 * a tab.
	 */
	class  coTUITab : public coTUIElement
	{

		

	private:
	public:
		coTUITab(const std::string&, int pID = 1);
		coTUITab(coTabletUI* tui, const std::string&, int pID);
		virtual ~coTUITab();
		virtual void parseMessage(TokenBuffer& tb) override;

	
		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();

	protected:
	};

	/**
	 * a dynamic UI tab.
	 */
	class  coTUIUITab : public coTUIElement
	{

		

	private:
	public:
		coTUIUITab(const std::string&, int pID = 1);
		coTUIUITab(coTabletUI* tui, const std::string&, int pID = 1);
		virtual ~coTUIUITab();
		virtual void parseMessage(TokenBuffer& tb) override;

		bool loadUIFile(const std::string& filename);

	
		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();


	private:
		std::string filename;
		std::string uiDescription;
	};

	/**
	 * a tab folder.
	 */
	class  coTUITabFolder : public coTUIElement
	{
		

	private:
	public:
		coTUITabFolder(const std::string&, int pID = 1);
		coTUITabFolder(coTabletUI* tui, const std::string&, int pID = 1);
		virtual ~coTUITabFolder();
		virtual void parseMessage(TokenBuffer& tb) override;

	
		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();

	protected:
	};

	class  coTUISGBrowserTab : public coTUIElement
	{
	private:
		std::string findName;
		int visitorMode;
		int polyMode;
		int selOnOff;
		int sendImageMode;

		float ColorR, ColorG, ColorB;
		std::string expandPath;
		std::string selectPath;
		std::string selectParentPath;
		std::string showhidePath;
		std::string showhideParentPath;

	public:
		float diffuse[4];
		float specular[4];
		float ambient[4];
		float emissive[4];
		float matrix[16];
		bool loadFile;

		coTUISGBrowserTab(const char*, int pID = 1);
		coTUISGBrowserTab(coTabletUI* tui, const char*, int pID = 1);
		virtual ~coTUISGBrowserTab();
		virtual void resend(bool create) override;

		int openServer();
		virtual void parseMessage(TokenBuffer& tb) override;
		virtual void sendType(int type, const char* nodeType, const char* name, const char* path, const char* pPath, int mode, int numChildren = 0);
		virtual void sendEnd();
		virtual void sendProperties(std::string path, std::string pPath, int mode, int transparent);
		virtual void sendProperties(std::string path, std::string pPath, int mode, int transparent, float mat[]);
		virtual void sendCurrentNode(osg::Node* node, std::string);
		virtual void sendRemoveNode(std::string path, std::string parentPath);
		virtual void sendShader(std::string name);
		virtual void sendUniform(std::string name, std::string type, std::string value, std::string min, std::string max, std::string textureFile);
		virtual void sendShaderSource(std::string vertex, std::string fragment, std::string geometry, std::string tessControl, std::string tessEval);
		virtual void updateUniform(std::string shader, std::string name, std::string value, std::string textureFile);
		virtual void updateShaderSourceV(std::string shader, std::string vertex);
		virtual void updateShaderSourceF(std::string shader, std::string fragment);
		virtual void updateShaderSourceG(std::string shader, std::string geometry);
		virtual void updateShaderSourceTE(std::string shader, std::string tessEval);
		virtual void updateShaderSourceTC(std::string shader, std::string tessControl);
		virtual void updateShaderNumVertices(std::string shader, int);
		virtual void updateShaderOutputType(std::string shader, int);
		virtual void updateShaderInputType(std::string shader, int);

		virtual const std::string& getFindName() const
		{
			return findName;
		}
		virtual int getVisMode() const
		{
			return visitorMode;
		}
		virtual int getImageMode() const
		{
			return sendImageMode;
		}
		virtual osg::Node* getCurrentNode()
		{
			return currentNode;
		}

		virtual const std::string& getExpandPath() const
		{
			return expandPath;
		}
		virtual const std::string& getSelectPath() const
		{
			return selectPath;
		}
		virtual const std::string& getSelectParentPath() const
		{
			return selectParentPath;
		}
		virtual const std::string& getShowHidePath() const
		{
			return showhidePath;
		}
		virtual const std::string& getShowHideParentPath() const
		{
			return showhideParentPath;
		}
		virtual float getR() const
		{
			return ColorR;
		}
		virtual float getG() const
		{
			return ColorG;
		}
		virtual float getB() const
		{
			return ColorB;
		}
		virtual int getPolyMode() const
		{
			return polyMode;
		}
		virtual int getSelMode() const
		{
			return selOnOff;
		}

		virtual void parseTextureMessage(TokenBuffer& tb);
		virtual void setTexture(int height, int width, int depth, int texIndex, int dataLength, const char* data);
		virtual void setTexture(int texNumber, int mode, int texGenMode, int texIndex);
		virtual void setCurrentNode(osg::Node* node)
		{
			currentNode = node;
		}
		virtual void setCurrentPath(std::string str)
		{
			currentPath = str;
		}
		virtual void decTexturesToChange()
		{
			if (texturesToChange > 0)
				--texturesToChange;
		}
		virtual void finishedTraversing();
		virtual void sendTraversedTextures();
		virtual void finishedNode();
		virtual void noTexture();
		virtual void sendNodeTextures();
		virtual void sendNoTextures();
		virtual void incTextureListCount();
		virtual void sendTexture();
		virtual void loadFilesFlag(bool state);
		virtual void hideSimNode(bool state, char* nodePath, char* parentPath);
		virtual void setSimPair(char* nodePath, char* simPath, char* simName);

		virtual int queueIsEmpty() const
		{
			return _dataList.empty();
		}
		virtual int getHeight() const
		{
			return height;
		}
		virtual int getWidth() const
		{
			return width;
		}
		virtual int getDepth() const
		{
			return depth;
		}
		virtual int getIndex() const
		{
			return index;
		}
		virtual size_t getDataLength() const;
		virtual int getTextureNumber() const
		{
			return textureNumber;
		}
		virtual int getTextureMode() const
		{
			return textureMode;
		}
		virtual int getTextureTexGenMode() const
		{
			return textureTexGenMode;
		}
		virtual int getTexturesToChange() const
		{
			return texturesToChange;
		}
		virtual int hasAlpha() const
		{
			return alpha;
		}
		virtual char* getData();
		virtual Connection* getConnection();
		virtual osg::Node* getChangedNode()
		{
			return changedNode;
		}
		virtual const std::string& getChangedPath() const
		{
			return changedPath;
		}

		void send(TokenBuffer& tb);
		void tryConnect();
		void parseTextureMessage();

	protected:
		virtual void lock()
		{
			mutex.lock();
		}
		virtual void unlock()
		{
			mutex.unlock();
		}
		int texturesToChange = 0;
		int height = 0;
		int width = 0;
		int depth = 0;
		std::vector<char> data;
		int textureNumber;

		int index;

		osg::Node* changedNode = nullptr;
		//Connection *conn = nullptr;
		ServerConnection* sConn = nullptr;

		std::queue<int> _heightList;
		std::queue<int> _widthList;
		std::queue<int> _depthList;
		std::queue<int> _indexList;
		std::queue<int> _lengthList;
		std::queue<const char*> _dataList;

		int textureMode;
		int textureTexGenMode;
		int alpha;

		Host* serverHost = nullptr;
		Host* localHost = nullptr;
		int texturePort;
		SGTextureThread* thread = nullptr;
		OpenThreads::Mutex mutex;

		osg::Node* currentNode = nullptr;
		std::string currentPath;
		std::string changedPath;
	};


	class  coTUIAnnotationTab : public coTUIElement
	{
	public:
		coTUIAnnotationTab(const char*, int pID = 1);
		virtual ~coTUIAnnotationTab();
		virtual void parseMessage(TokenBuffer& tb) override;

		void setNewButtonState(bool state);
		void addAnnotation(int id);
		void deleteAnnotation(int mode, int id);
		void setSelectedAnnotation(int id);
	};

	/**
	 * a NavigationElement.
	 */
	class  coTUINav : public coTUIElement
	{
	private:
	public:
		coTUINav(const char*, int pID = 1);
		virtual ~coTUINav();
		virtual void parseMessage(TokenBuffer& tb) override;
		bool down;
		int x;
		int y;

	protected:
	};
	/**
	 * a Splitter.
	 */
	class  coTUISplitter : public coTUIElement
	{
		

	private:
	public:
		enum orientations
		{
			Horizontal = 0x1,
			Vertical = 0x2
		};

		coTUISplitter(const std::string&, int pID = 1);
		virtual ~coTUISplitter();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void setShape(int s);
		virtual void setStyle(int t);
		virtual void setOrientation(int orientation);
		virtual int getShape() const
		{
			return this->shape;
		}
		virtual int getStyle() const
		{
			return this->style;
		}
		virtual int getOrientation() const
		{
			return this->orientation;
		}

	
		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();

	protected:
		int shape;
		int style;
		int orientation;
	};

	/**
	 * a Frame.
	 */
	class  coTUIFrame : public coTUIElement
	{

		


	private:
	public:
		enum styles
		{
			Plain = 0x0010,
			Raised = 0x0020,
			Sunken = 0x0030
		};
		enum shapes
		{
			NoFrame = 0x0000,
			Box = 0x0001,
			Panel = 0x0002,
			WinPanel = 0x0003,
			HLine = 0x0004,
			VLine = 0x0005,
			StyledPanel = 0x0006
		};

		coTUIFrame(const std::string&, int pID = 1);
		coTUIFrame(coTabletUI* tui, const std::string&, int pID = 1);
		virtual ~coTUIFrame();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void setShape(int s); /* set shape first */
		virtual void setStyle(int t);

		virtual int getShape() const
		{
			return this->shape;
		}
		virtual int getStyle() const
		{
			return this->style;
		}

	
		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();

	protected:
		int style;
		int shape;
	};

	/**
	 * a GroupBox.
	 */
	class  coTUIGroupBox : public coTUIElement
	{

		

	private:
	public:
		coTUIGroupBox(const std::string&, int pID = 1);
		coTUIGroupBox(coTabletUI* tui, const std::string&, int pID = 1);
		virtual ~coTUIGroupBox();
		virtual void parseMessage(TokenBuffer& tb) override;

	public :

	
		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();
	};

	/**
	 * a toggle button.
	 */
	class  coTUIToggleButton : public coTUIElement
	{
		


	private:
	public:
		coTUIToggleButton(const std::string&, int pID = 1, bool state = false);
		coTUIToggleButton(coTabletUI* tui, const std::string&, int pID = 1, bool state = false);
		virtual ~coTUIToggleButton();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void setState(bool s);
		virtual bool getState() const;

	
		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();

	protected:
		bool state;
	};
	/**
	 * a toggleBitmapButton.
	 */
	class  coTUIToggleBitmapButton : public coTUIElement
	{

		

	private:
	public:
		coTUIToggleBitmapButton(const std::string&, const std::string&, int pID = 1, bool state = false);
		virtual ~coTUIToggleBitmapButton();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void setState(bool s);
		virtual bool getState() const;

	
		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();

	protected:
		bool state;
		std::string bmpUp;
		std::string bmpDown;
	};
	/**
	 * a messageBox.
	 */
	class  coTUIMessageBox : public coTUIElement
	{

		

	private:
	public:
		coTUIMessageBox(const std::string&, int pID = 1);
		virtual ~coTUIMessageBox();

	protected:
	};
	/**
	 * a ProgressBar.
	 */
	class  coTUIProgressBar : public coTUIElement
	{

		

	private:
	public:
		coTUIProgressBar(const std::string&, int pID = 1);
		virtual ~coTUIProgressBar();
		virtual void resend(bool create) override;

	public :
		virtual void setValue(int newV);
		virtual void setMax(int maxV);
		virtual int getValue() const
		{
			return this->actValue;
		}
		virtual int getMax() const
		{
			return this->maxValue;
		}

	protected:
		int actValue;
		int maxValue;
	};
	/**
	 * a slider.
	 */
	class  coTUIFloatSlider : public coTUIElement
	{

		


	private:
	public:
		enum Orientation
		{
			HORIZONTAL = 1,
			VERTICAL = 0
		};

		coTUIFloatSlider(const std::string&, int pID = 1, bool state = true);
		coTUIFloatSlider(coTabletUI* tui, const std::string&, int pID = 1, bool state = true);
		virtual ~coTUIFloatSlider();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void setValue(float newV);
		virtual void setTicks(int t);
		virtual void setOrientation(bool);
		virtual void setMin(float minV);
		virtual void setMax(float maxV);
		virtual void setRange(float minV, float maxV);
		virtual void setLogarithmic(bool val);

		virtual float getValue() const
		{
			return this->actValue;
		}
		virtual int getTicks() const
		{
			return this->ticks;
		}
		virtual bool getOrientation() const
		{
			return this->orientation;
		}
		virtual float getMin() const
		{
			return this->minValue;
		}
		virtual float getMax() const
		{
			return this->maxValue;
		}
		virtual bool getLogarithmic() const
		{
			return this->logarithmic;
		}
	
		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();

	protected:
		float actValue;
		float minValue;
		float maxValue;
		int ticks;
		bool orientation;
		bool logarithmic = false;
	};
	/**
	 * a slider.
	 */
	class  coTUISlider : public coTUIElement
	{
		

	private:
	public:
		enum Orientation
		{
			HORIZONTAL = 1,
			VERTICAL = 0
		};

		coTUISlider(const std::string&, int pID = 1, bool state = true);
		coTUISlider(coTabletUI* tui, const std::string&, int pID = 1, bool state = true);
		virtual ~coTUISlider();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void setValue(int newV);
		virtual void setOrientation(bool o);
		virtual void setTicks(int t);
		virtual void setMin(int minV);
		virtual void setMax(int maxV);
		virtual void setRange(int minV, int maxV);

		virtual int getValue() const
		{
			return this->actValue;
		}
		virtual int getTicks() const
		{
			return this->ticks;
		}
		virtual bool getOrientation() const
		{
			return this->orientation;
		}
		virtual int getMin() const
		{
			return this->minValue;
		}
		virtual int getMax() const
		{
			return this->maxValue;
		}

	
		void tabletEvent();
		void tabletPressEvent();
		void tabletReleaseEvent();

	protected:
		int actValue;
		int minValue;
		int maxValue;
		int ticks;
		bool orientation;
	};
	/**
	 * a spinEditField.
	 */
	class  coTUISpinEditfield : public coTUIElement
	{

		

	private:
	public:
		coTUISpinEditfield(const std::string&, int pID = 1);
		virtual ~coTUISpinEditfield();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void setPosition(int newV);
		virtual void setMin(int minV);
		virtual void setMax(int maxV);
		virtual void setStep(int s);

		virtual int getValue() const
		{
			return this->actValue;
		}
		virtual int getStep() const
		{
			return this->step;
		}
		virtual int getMin() const
		{
			return this->minValue;
		}
		virtual int getMax() const
		{
			return this->maxValue;
		}

	
		void tabletEvent();

	protected:
		int actValue;
		int minValue;
		int maxValue;
		int step;
	};
	/**
	 * a spinEditField with text.
	 */
	class  coTUITextSpinEditField : public coTUIElement
	{

		


	private:
	public:
		coTUITextSpinEditField(const std::string&, int pID = 1);
		virtual ~coTUITextSpinEditField();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void setMin(int minV);
		virtual void setMax(int maxV);
		virtual void setStep(int s);
		virtual void setText(const std::string& text);
		virtual const std::string& getText() const
		{
			return this->text;
		}
		virtual int getStep() const
		{
			return this->step;
		}
		virtual int getMin() const
		{
			return this->minValue;
		}
		virtual int getMax() const
		{
			return this->maxValue;
		}

	
		void tabletEvent();

	protected:
		std::string text;
		int minValue;
		int maxValue;
		int step;
	};
	/**
	 * a editField. (LineEdit)
	 */
	class  coTUIEditField : public coTUIElement
	{

		


	private:
	public:
		coTUIEditField(const std::string&, int pID = 1);
		coTUIEditField(coTabletUI* tui, const std::string&, int pID = 1);
		virtual ~coTUIEditField();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void setText(const std::string& t);
		virtual void setImmediate(bool);
		virtual void setPasswordMode(bool b);
		virtual void setIPAddressMode(bool b);

		virtual const std::string& getText() const;
		virtual bool isImmediate() const
		{
			return this->immediate;
		}

	
		void tabletEvent();

	protected:
		std::string text;
		bool immediate;
	};
	/**
	 * another editField (TextEdit)
	 */
	class  coTUIEditTextField : public coTUIElement
	{
		


	private:
	public:
		coTUIEditTextField(const std::string&, int pID = 1);
		coTUIEditTextField(coTabletUI* tui, const std::string&, int pID = 1);
		virtual ~coTUIEditTextField();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void setText(const std::string& t);
		virtual void setImmediate(bool);
		virtual const std::string& getText() const;
		virtual bool isImmediate() const
		{
			return this->immediate;
		}

	
		void tabletEvent();

	protected:
		std::string text;
		bool immediate;
	};
	/**
	 * a editIntField = EditField fuer Integer
	 */
	class  coTUIEditIntField : public coTUIElement
	{
		


	private:
	public:
		coTUIEditIntField(const std::string&, int pID = 1, int def = 0);
		coTUIEditIntField(coTabletUI* tui, const std::string&, int pID = 1, int def = 0);
		virtual ~coTUIEditIntField();
		virtual void parseMessage(TokenBuffer& tb) override;
		virtual void resend(bool create) override;
		virtual std::string getText() const;

	public :
		virtual void setImmediate(bool);
		virtual void setValue(int val);
		virtual void setMin(int min);
		virtual void setMax(int max);
		virtual int getValue() const
		{
			return this->value;
		}
		virtual bool isImmediate() const
		{
			return this->immediate;
		}
		virtual int getMin() const
		{
			return this->min;
		}
		virtual int getMax() const
		{
			return this->max;
		}

	
		void tabletEvent();

	protected:
		int value;
		int min;
		int max;
		bool immediate;
	};
	/**
	 * a editfloatfield = EditField fuer Kommazahlen
	 */
	class  coTUIEditFloatField : public coTUIElement
	{
		


	private:
	public:
		coTUIEditFloatField(const std::string&, int pID = 1, float def = 0);
		coTUIEditFloatField(coTabletUI* tui, const std::string&, int pID = 1, float def = 0);
		virtual ~coTUIEditFloatField();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void setImmediate(bool);
		virtual void setValue(float val);
		virtual float getValue() const
		{
			return this->value;
		}
		virtual bool isImmediate() const
		{
			return this->immediate;
		}
	
		void tabletEvent();

	protected:
		float value;
		bool immediate;
	};
	/**
	 * a comboBox.
	 */
	class  coTUIComboBox : public coTUIElement
	{

		

	private:
	public:
		coTUIComboBox(const std::string&, int pID = 1);
		coTUIComboBox(coTabletUI* tui, const std::string&, int pID = 1);
		virtual ~coTUIComboBox();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void addEntry(const std::string& t);
		virtual void delEntry(const std::string& t);
		virtual void clear();
		virtual int getSelectedEntry() const;
		virtual void setSelectedEntry(int e);
		virtual void setSelectedText(const std::string& t);
		virtual const std::string& getSelectedText() const;
		virtual int getNumEntries();

	
		void tabletEvent();

	protected:
		std::string text;
		int selection;
		std::list<std::string> elements;
	};
	/**
	 * a listBox.
	 */
	class  coTUIListBox : public coTUIElement
	{
		

	private:
	public:
		coTUIListBox(const std::string&, int pID = 1);
		virtual ~coTUIListBox();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	public :
		virtual void addEntry(const std::string& t);
		virtual void delEntry(const std::string& t);
		virtual int getSelectedEntry() const;
		virtual void setSelectedEntry(int e);
		virtual void setSelectedText(const std::string& t);
		virtual const std::string& getSelectedText() const;

	
		void tabletEvent();

	protected:
		std::string text;
		int selection;
		std::list<std::string> elements;
	};
	class  MapData
	{
	public:
		MapData(const char* name, float ox, float oy, float xSize, float ySize, float height);
		virtual ~MapData();
		char* name;
		float ox, oy, xSize, ySize, height;
	};
	/**
	 * a Map Widget
	 */
	class  coTUIMap : public coTUIElement
	{
	private:
	public:
		coTUIMap(const char*, int pID = 1);
		virtual ~coTUIMap();
		virtual void addMap(const char* name, float ox, float oy, float xSize, float ySize, float height);
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

		float angle;
		float xPos;
		float yPos;
		float height;
		int mapNum;

	protected:
		std::list<MapData*> maps;
	};
	/**
	* an earth Map Widget
	*/
	class  coTUIEarthMap : public coTUIElement
	{
	private:
	public:
		coTUIEarthMap(const char*, int pID = 1);
		virtual ~coTUIEarthMap();
		virtual void setPosition(float latitude, float longitude, float altitude);
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;


		float latitude;
		float longitude;
		float altitude;
		float minHeight;
		float maxHeight;

		void addPathNode(float latitude, float longitude);
		std::list<std::pair<float, float>> path;
		void updatePath();
		void setMinMax(float minH, float maxH);


	protected:
	};
	/**
	 * PopUp Window with text
	 */
	class  coTUIPopUp : public coTUIElement
	{


	private:
	public:
		coTUIPopUp(const std::string&, int pID = 1);
		virtual ~coTUIPopUp();
		virtual void resend(bool create) override;
		virtual void parseMessage(TokenBuffer& tb) override;

	
		void tabletEvent();

	public :
		virtual void setText(const std::string& t);
		virtual const std::string& getText() const
		{
			return this->text;
		}
		virtual void setImmediate(bool);
		virtual bool isImmediate() const
		{
			return this->immediate;
		}

	protected:
		std::string text;
		bool immediate;
	};
#endif
