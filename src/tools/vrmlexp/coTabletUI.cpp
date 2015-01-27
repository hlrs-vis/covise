/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
#include <coTabletUIMessages.h>
#include "coTabletUI.h"

coTUIButton::coTUIButton(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_BUTTON)
{
}

coTUIButton::~coTUIButton()
{
}

void coTUIButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_PRESSED)
    {
        if (listener)
            listener->tabletPressEvent(this);
    }
    else if (i == TABLET_RELEASED)
    {
        if (listener)
            listener->tabletReleaseEvent(this);
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIButton::resend()
{
    createSimple(TABLET_BUTTON);
    coTUIElement::resend();
}
coTUIColorTriangle::coTUIColorTriangle(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_COLOR_TRIANGLE)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
}

coTUIColorTriangle::~coTUIColorTriangle()
{
}

void coTUIColorTriangle::parseMessage(TokenBuffer &tb)
{
    int i, j;
    tb >> i;
    tb >> j;

    if (i == TABLET_RGBA)
    {
        tb >> red;
        tb >> green;
        tb >> blue;

        if (j == TABLET_RELEASED)
        {
            if (listener)
                listener->tabletReleaseEvent(this);
        }
        if (j == TABLET_PRESSED)
        {
            if (listener)
                listener->tabletEvent(this);
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIColorTriangle::setColor(float r, float g, float b)
{
    red = r;
    green = g;
    blue = b;
    setVal(TABLET_RED, r);
    setVal(TABLET_GREEN, g);
    setVal(TABLET_BLUE, b);
}

void coTUIColorTriangle::resend()
{
    createSimple(TABLET_COLOR_TRIANGLE);
    setVal(TABLET_RED, red);
    setVal(TABLET_GREEN, green);
    setVal(TABLET_BLUE, blue);
    coTUIElement::resend();
}

coTUIColorTab::coTUIColorTab(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_COLOR_TAB)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
    alpha = 1.0;
}

coTUIColorTab::~coTUIColorTab()
{
}

void coTUIColorTab::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_RGBA)
    {
        tb >> red;
        tb >> green;
        tb >> blue;
        tb >> alpha;

        if (listener)
            listener->tabletEvent(this);
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIColorTab::setColor(float r, float g, float b, float a)
{
    red = r;
    green = g;
    blue = b;
    alpha = a;

    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_RGBA;
    t << ID;
    t << r;
    t << g;
    t << b;
    t << a;
    coTabletUI::tUI->send(t);
}

void coTUIColorTab::resend()
{
    createSimple(TABLET_COLOR_TAB);
    setColor(red, green, blue, alpha);
    coTUIElement::resend();
}

coTUINav::coTUINav(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_NAV_ELEMENT)
{
    down = false;
    x = 0;
    y = 0;
}

coTUINav::~coTUINav()
{
}

void coTUINav::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_PRESSED)
    {
        tb >> x;
        tb >> y;
        if (listener)
            listener->tabletPressEvent(this);
        down = true;
    }
    else if (i == TABLET_RELEASED)
    {
        tb >> x;
        tb >> y;
        if (listener)
            listener->tabletReleaseEvent(this);
        down = false;
    }
    else if (i == TABLET_POS)
    {
        tb >> x;
        tb >> y;
        if (listener)
            listener->tabletEvent(this);
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUINav::resend()
{
    createSimple(TABLET_NAV_ELEMENT);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIBitmapButton::coTUIBitmapButton(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_BITMAP_BUTTON)
{
}

coTUIBitmapButton::~coTUIBitmapButton()
{
}

void coTUIBitmapButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_PRESSED)
    {
        if (listener)
            listener->tabletPressEvent(this);
    }
    else if (i == TABLET_RELEASED)
    {
        if (listener)
            listener->tabletReleaseEvent(this);
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIBitmapButton::resend()
{
    createSimple(TABLET_BITMAP_BUTTON);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUILabel::coTUILabel(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_TEXT_FIELD)
{
}

coTUILabel::~coTUILabel()
{
}

void coTUILabel::resend()
{

    createSimple(TABLET_TEXT_FIELD);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUITabFolder::coTUITabFolder(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_TAB_FOLDER)
{
}

coTUITabFolder::~coTUITabFolder()
{
}

void coTUITabFolder::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
        if (listener)
            listener->tabletPressEvent(this);
    }
    else if (i == TABLET_DISACTIVATED)
    {
        if (listener)
            listener->tabletReleaseEvent(this);
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUITabFolder::resend()
{
    createSimple(TABLET_TAB_FOLDER);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUITab::coTUITab(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_TAB)
{
}

coTUITab::~coTUITab()
{
}

void coTUITab::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
        if (listener)
            listener->tabletPressEvent(this);
    }
    else if (i == TABLET_DISACTIVATED)
    {
        if (listener)
            listener->tabletReleaseEvent(this);
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUITab::resend()
{
    createSimple(TABLET_TAB);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------
/*
void TextureThread::sleep(int msec)
{
#ifdef WIN32
			::Sleep(msec);
#else
			usleep(msec*1000);
#endif
}

void TextureThread::run()
{
	while(running)
	{
		while(!tab->getConnection())
		{
			this->
			tab->tryConnect();
			sleep(250);

		}
		//if(type == TABLET_TEX_UPDATE)
		{	int count = 0;
			while(!tab->queueIsEmpty())
			{
				tab->sendTexture();
				sendedTextures++;

				//cout << " sended textures : " << sendedTextures << "\n";
				if(finishedTraversing && textureListCount == sendedTextures)
				{
					//cout << " finished sending with: " << textureListCount << "  textures \n";
					tab->sendTraversedTextures();
					//type = THREAD_NOTHING_TO_DO;
					finishedTraversing = false;
					textureListCount = 0;
					sendedTextures = 0;
				}
				if(count++ == 5) break;
			}
			//sleep(100);
		}
		if(tab->getTexturesToChange()) //still Textures in queue
		{
			int count = 0;
			Message m;
			// waiting for incoming data
			while(count<200)
			{
				count++;

				if(tab->getConnection())
				{
					if(tab->getConnection()->check_for_input()) // message arrived
					{
						tab->getConnection()->recv_msg(&m);
						TokenBuffer tokenbuffer(&m);
						if(m.type == TABLET_UI)
						{
							int ID;
							tokenbuffer >> ID;
							tab->parseTextureMessage(tokenbuffer);
							tab->decTexturesToChange();
							//type = THREAD_NOTHING_TO_DO;
							break;
						}
					}
				}
			sleep(50);
			}
		}
		else sleep(50);
	}
}

coTUITextureTab::coTUITextureTab(const char *n, int pID):coTUIElement(n,pID,TABLET_TEXTURE_TAB)
{
	conn = NULL;
	currentNode = 0;
        changedNode = 0;
	texturesToChange = 0;
	thread = new TextureThread(this);
	thread->setType(THREAD_NOTHING_TO_DO);
	thread->start();
}


coTUITextureTab::~coTUITextureTab()
{
	thread->terminateTextureThread();

	while(thread->isRunning())
	{
#ifdef WIN32
		::Sleep(1000);
#else
		usleep(1000000);
#endif
	}
	delete conn;
	conn = NULL;
}
void coTUITextureTab::finishedTraversing()
{
	thread->traversingFinished();
}

void coTUITextureTab::incTextureListCount()
{
	thread->incTextureListCount();
}

void coTUITextureTab::sendTraversedTextures()
{
	TokenBuffer t;
	t << TABLET_SET_VALUE;
	t << TABLET_TRAVERSED_TEXTURES;
	t << ID;
	this->send(t);
}

void coTUITextureTab::parseMessage(TokenBuffer &tb)
{
	int i;
	tb >> i;
	if(i == TABLET_TEX_UPDATE)
	{
		thread->setType(TABLET_TEX_UPDATE);
		if(listener)
			listener->tabletReleaseEvent(this);
	}
	else if(i == TABLET_TEX_CHANGE)
	{
		//cout << " currentNode : " << currentNode << "\n";
		if(currentNode)
		{
			// let the tabletui know that it can send texture data now
			TokenBuffer t;
			int buttonNumber;
			tb >> buttonNumber;
			t << TABLET_SET_VALUE;
			t << TABLET_TEX_CHANGE;
			t << ID;
			t << buttonNumber;
			t << (uint64_t)(uintptr_t)currentNode;
			coTabletUI::tUI->send(t);
			//thread->setType(TABLET_TEX_CHANGE);
			texturesToChange++;
		}
	}
	else
	{
		cerr << "unknown event "<< i << endl;
	}
}

void coTUITextureTab::parseTextureMessage(TokenBuffer &tb)
{
	int type;
	tb >> type;
	if(type == TABLET_TEX_CHANGE)
	{
        uint64_t node;
		tb >> textureNumber;
		tb >> textureMode;
		tb >> textureTexGenMode;
		tb >> alpha;
		tb >> height;
		tb >> width;
		tb >> depth;
		tb >> dataLength;
		tb >> node;

        changedNode = (osg::Node *)(uintptr_t)node;
		if(changedNode)
		{
			data = new char[dataLength];

			for(int k = 0; k < dataLength; k++)
			{
				if( (k%4) == 3)
					tb >> data[k];
				else if( (k%4) == 2)
					tb >> data[k-2];
				else if( (k%4) == 1)
					tb >> data[k];
				else if( (k%4) == 0)
					tb >> data[k+2];
			}
			if(listener)
				listener->tabletPressEvent(this);
		}
	}
}
void coTUITextureTab::sendTexture()
{
	mutex.lock();
	TokenBuffer tb;
	tb << TABLET_SET_VALUE;
	tb << TABLET_TEX;
	tb << ID;
	tb << heightList.front();
	tb << widthList.front();
	tb << depthList.front();
	tb << lengthList.front();

	int length = heightList.front() * widthList.front()*depthList.front()/8;
    tb.addBinary(dataList.front(),length);
	this->send(tb);
	heightList.pop();
	widthList.pop();
	depthList.pop();
	lengthList.pop();
	dataList.pop();
	mutex.unlock();
}

void coTUITextureTab::setTexture(int height, int width, int depth, int dataLength, const char* data)
{
	mutex.lock();
	heightList.push(height);
	widthList.push(width);
	depthList.push(depth);
	lengthList.push(dataLength);
	dataList.push(data);
	thread->incTextureListCount();
	//cout << " added texture : \n";
	mutex.unlock();
}

void coTUITextureTab::setTexture(int texNumber, int mode, int texGenMode)
{
	if(coTabletUI::tUI->conn==NULL)
		return;

	TokenBuffer tb;
	tb << TABLET_SET_VALUE;
	tb << TABLET_TEX_MODE;
	tb << ID;
	tb << texNumber;
	tb << mode;
	tb << texGenMode;

	coTabletUI::tUI->send(tb);
}


void coTUITextureTab::resend()
{
	createSimple(TABLET_TEXTURE_TAB);
	coTUIElement::resend();
}

void coTUITextureTab::tryConnect()
{
	serverHost = NULL;
	localHost = new Host("localhost");
	const char *line;
	port = 31810;
	coCoviseConfig::getEntry ("TabletPC.TCPPort", &port);
	line = coCoviseConfig::getEntry ("TabletPC.Server");

	timeout = 0.0;
	coCoviseConfig::getEntry("TabletPC.Timeout", &timeout);
	if (line)
	{
		if(strcasecmp(line,"NONE")==0)
			serverHost = NULL;
		else
			serverHost = new Host(line);
	}
	else
	{
		serverHost = NULL;
	}
	conn = new ClientConnection(serverHost,port,0,(sender_type)0,0);
	if(!conn->is_connected())             // could not open server port
	{
#ifndef _WIN32
      if(errno!=ECONNREFUSED)
      {
         fprintf(stderr,"Could not connect to TabletPC %s; port %d: %s\n",
               localHost->get_name(),port, strerror(errno));
      }
#else
      fprintf(stderr,"Could not connect to TabletPC %s; port %d\n",localHost->get_name(),port);
#endif
		delete conn;
		conn=NULL;

		conn = new ClientConnection(localHost,port,0,(sender_type)0,0);
		if(!conn->is_connected())             // could not open server port
		{
	#ifndef _WIN32
			if(errno!=ECONNREFUSED)
			{
				fprintf(stderr,"Could not connect to TabletPC %s; port %d: %s\n",
					localHost->get_name(),port, strerror(errno));
			}
	#else
			fprintf(stderr,"Could not connect to TabletPC %s; port %d\n",localHost->get_name(),port);
	#endif
			delete conn;
			conn=NULL;
		}
	}
}

void coTUITextureTab::send(TokenBuffer &tb)
{
	if(conn==NULL)
		return;
	Message m(tb);
	m.type =TABLET_UI;
	conn->send_msg(&m);
}
//----------------------------------------------------------
//----------------------------------------------------------
*/
coTUISGBrowserTab::coTUISGBrowserTab(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_BROWSER_TAB)
{
}

coTUISGBrowserTab::~coTUISGBrowserTab()
{
}

void coTUISGBrowserTab::sendType(int type, const char *nodeType, const char *name, osg::Node *parent, osg::Node *nodeptr)
{
    if (coTabletUI::tUI->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_NODE;
    tb << ID;
    tb << type;
    tb << name;
    tb << (uint64_t)(uintptr_t)nodeptr;
    tb << (uint64_t)(uintptr_t)parent;
    tb << nodeType;

    coTabletUI::tUI->send(tb);
}

void coTUISGBrowserTab::sendEnd()
{
    if (coTabletUI::tUI->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_END;
    tb << ID;
    coTabletUI::tUI->send(tb);
}
void coTUISGBrowserTab::sendCurrentNode(osg::Node *node)
{
    currentNode = node;
    if (coTabletUI::tUI->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_CURRENT_NODE;
    tb << ID;
    tb << (uint64_t)(uintptr_t)currentNode;
    //cerr << "current PTR: "<< currentNode << endl;
    coTabletUI::tUI->send(tb);
}

void coTUISGBrowserTab::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_BROWSER_UPDATE)
    {
        if (listener)
            listener->tabletPressEvent(this);
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUISGBrowserTab::resend()
{
    createSimple(TABLET_BROWSER_TAB);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUISplitter::coTUISplitter(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_SPLITTER)
{
    shape = coTUIFrame::StyledPanel;
    style = coTUIFrame::Sunken;
    setShape(shape);
    setStyle(style);
    setOrientation(orientation);
}

coTUISplitter::~coTUISplitter()
{
}

void coTUISplitter::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUISplitter::resend()
{
    createSimple(TABLET_SPLITTER);
    coTUIElement::resend();
    setShape(shape);
    setStyle(style);
    setOrientation(orientation);
}

void coTUISplitter::setShape(int s)
{
    TokenBuffer tb;
    shape = s;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SHAPE;
    tb << ID;
    tb << shape;
    coTabletUI::tUI->send(tb);
}

void coTUISplitter::setStyle(int t)
{
    TokenBuffer tb;
    style = t;
    tb << TABLET_SET_VALUE;
    tb << TABLET_STYLE;
    tb << ID;
    tb << (style | shape);
    coTabletUI::tUI->send(tb);
}

void coTUISplitter::setOrientation(int or )
{
    TokenBuffer tb;
    orientation = or ;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ORIENTATION;
    tb << ID;
    tb << orientation;
    coTabletUI::tUI->send(tb);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIFrame::coTUIFrame(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_FRAME)
{
    style = Sunken;
    shape = StyledPanel;
    setShape(shape);
    setStyle(style);
}

coTUIFrame::~coTUIFrame()
{
}

void coTUIFrame::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIFrame::setStyle(int t)
{
    TokenBuffer tb;
    style = t;
    tb << TABLET_SET_VALUE;
    tb << TABLET_STYLE;
    tb << ID;
    tb << (style | shape);
    coTabletUI::tUI->send(tb);
}

void coTUIFrame::setShape(int s)
{
    TokenBuffer tb;
    shape = s;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SHAPE;
    tb << ID;
    tb << shape;
    coTabletUI::tUI->send(tb);
}

void coTUIFrame::resend()
{
    createSimple(TABLET_FRAME);

    coTUIElement::resend();
    setShape(shape);
    setStyle(style);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIToggleButton::coTUIToggleButton(const char *n, int pID, bool s)
    : coTUIElement(n, pID, TABLET_TOGGLE_BUTTON)
{
    state = s;
    setVal(state);
}

coTUIToggleButton::~coTUIToggleButton()
{
}

void coTUIToggleButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
        state = true;
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
        state = false;
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIToggleButton::setState(bool s)
{
    if (s != state) // don�t send unnecessary state changes
    {
        state = s;
        setVal(state);
    }
}

bool coTUIToggleButton::getState()
{
    return state;
}

void coTUIToggleButton::resend()
{
    createSimple(TABLET_TOGGLE_BUTTON);
    setVal(state);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIToggleBitmapButton::coTUIToggleBitmapButton(const char *n, const char *down, int pID, bool state)
    : coTUIElement(n, pID, TABLET_BITMAP_TOGGLE_BUTTON)
{
    bmpUp = new char[strlen(n) + 1];
    strcpy(bmpUp, n);
    bmpDown = new char[strlen(down) + 1];
    strcpy(bmpDown, down);

    setVal(bmpDown);
    setVal(state);
}

coTUIToggleBitmapButton::~coTUIToggleBitmapButton()
{
}

void coTUIToggleBitmapButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIToggleBitmapButton::setState(bool s)
{
    if (s != state) // don�t send unnecessary state changes
    {
        state = s;
        setVal(state);
    }
}

bool coTUIToggleBitmapButton::getState()
{
    return state;
}

void coTUIToggleBitmapButton::resend()
{
    createSimple(TABLET_BITMAP_TOGGLE_BUTTON);
    setVal(bmpDown);
    setVal(state);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIMessageBox::coTUIMessageBox(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_MESSAGE_BOX)
{
}

coTUIMessageBox::~coTUIMessageBox()
{
}

void coTUIMessageBox::resend()
{
    createSimple(TABLET_MESSAGE_BOX);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIEditField::coTUIEditField(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_EDIT_FIELD)
{
    if (name)
    {
        text = new char[strlen(name) + 1];
        strcpy(text, name);
        setVal(text);
    }
    else
        text = 0;
    immediate = false;
}

coTUIEditField::~coTUIEditField()
{
    delete[] text;
}

void coTUIEditField::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void coTUIEditField::parseMessage(TokenBuffer &tb)
{
    char *m;
    tb >> m;
    delete[] text;
    text = new char[strlen(m) + 1];
    strcpy(text, m);
    if (listener)
        listener->tabletEvent(this);
}

void coTUIEditField::setPasswordMode(bool b)
{
    setVal(TABLET_ECHOMODE, (int)b);
}

void coTUIEditField::setIPAddressMode(bool b)
{
    setVal(TABLET_IPADDRESS, (int)b);
}

void coTUIEditField::setText(const char *t)
{
    delete[] text;
    if (t)
    {
        text = new char[strlen(t) + 1];
        strcpy(text, t);
        setVal(text);
    }
    else
        text = 0;
}

const char *coTUIEditField::getText()
{
    return text;
}

void coTUIEditField::resend()
{
    createSimple(TABLET_EDIT_FIELD);
    if (text)
        setVal(text);
    setVal(immediate);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIEditIntField::coTUIEditIntField(const char *n, int pID, int def)
    : coTUIElement(n, pID, TABLET_INT_EDIT_FIELD)
{
    value = def;
    immediate = 0;
    setVal(value);
}

coTUIEditIntField::~coTUIEditIntField()
{
}

void coTUIEditIntField::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void coTUIEditIntField::parseMessage(TokenBuffer &tb)
{
    tb >> value;
    if (listener)
        listener->tabletEvent(this);
}

const char *coTUIEditIntField::getText()
{
    return NULL;
}

void coTUIEditIntField::setMin(int min)
{
    //cerr << "coTUIEditIntField::setMin " << min << endl;
    this->min = min;
    setVal(TABLET_MIN, min);
}

void coTUIEditIntField::setMax(int max)
{
    //cerr << "coTUIEditIntField::setMax " << max << endl;
    this->max = max;
    setVal(TABLET_MAX, max);
}

void coTUIEditIntField::setValue(int val)
{
    if (value != val)
    {
        value = val;
        setVal(value);
    }
}

void coTUIEditIntField::resend()
{
    createSimple(TABLET_INT_EDIT_FIELD);
    setVal(TABLET_MIN, min);
    setVal(TABLET_MAX, max);
    setVal(value);
    setVal(immediate);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIEditFloatField::coTUIEditFloatField(const char *n, int pID, float def)
    : coTUIElement(n, pID, TABLET_FLOAT_EDIT_FIELD)
{
    value = def;
    setVal(value);
    immediate = 0;
}

coTUIEditFloatField::~coTUIEditFloatField()
{
}

void coTUIEditFloatField::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void coTUIEditFloatField::parseMessage(TokenBuffer &tb)
{
    tb >> value;
    if (listener)
        listener->tabletEvent(this);
}

void coTUIEditFloatField::setValue(float val)
{
    if (value != val)
    {
        value = val;
        setVal(value);
    }
}

void coTUIEditFloatField::resend()
{
    createSimple(TABLET_FLOAT_EDIT_FIELD);
    setVal(value);
    setVal(immediate);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUISpinEditfield::coTUISpinEditfield(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_SPIN_EDIT_FIELD)
{
    actValue = 0;
    minValue = 0;
    maxValue = 100;
    step = 1;
    setVal(actValue);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
}

coTUISpinEditfield::~coTUISpinEditfield()
{
}

void coTUISpinEditfield::parseMessage(TokenBuffer &tb)
{
    tb >> actValue;
    if (listener)
        listener->tabletEvent(this);
}

void coTUISpinEditfield::setPosition(int newV)
{

    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void coTUISpinEditfield::setStep(int newV)
{
    step = newV;
    setVal(TABLET_STEP, step);
}

void coTUISpinEditfield::setMin(int minV)
{
    minValue = minV;
    setVal(TABLET_MIN, minValue);
}

void coTUISpinEditfield::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void coTUISpinEditfield::resend()
{
    createSimple(TABLET_SPIN_EDIT_FIELD);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
    setVal(actValue);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------
coTUITextSpinEditField::coTUITextSpinEditField(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_TEXT_SPIN_EDIT_FIELD)
{
    text = 0;
    minValue = 0;
    maxValue = 100;
    step = 1;
    setVal(text);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
}

coTUITextSpinEditField::~coTUITextSpinEditField()
{
}

void coTUITextSpinEditField::parseMessage(TokenBuffer &tb)
{
    char *m;
    tb >> m;
    delete[] text;
    text = new char[strlen(m) + 1];
    strcpy(text, m);
    if (listener)
        listener->tabletEvent(this);
}

void coTUITextSpinEditField::setStep(int newV)
{
    step = newV;
    setVal(TABLET_STEP, step);
}

void coTUITextSpinEditField::setMin(int minV)
{
    minValue = minV;
    setVal(TABLET_MIN, minValue);
}

void coTUITextSpinEditField::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void coTUITextSpinEditField::resend()
{
    createSimple(TABLET_TEXT_SPIN_EDIT_FIELD);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
    setVal(text);
    coTUIElement::resend();
}

void coTUITextSpinEditField::setText(const char *t)
{
    cerr << "coTUITextSpinEditField::setText info: " << (t ? t : "*NULL*") << endl;
    delete[] text;
    if (t)
    {
        text = new char[strlen(t) + 1];
        strcpy(text, t);
        setVal(text);
    }
    else
        text = 0;
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIProgressBar::coTUIProgressBar(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_PROGRESS_BAR)
{
    actValue = 0;
    maxValue = 1;
}

coTUIProgressBar::~coTUIProgressBar()
{
}

void coTUIProgressBar::setValue(int newV)
{
    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void coTUIProgressBar::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void coTUIProgressBar::resend()
{
    createSimple(TABLET_PROGRESS_BAR);
    setVal(TABLET_MAX, maxValue);
    setVal(actValue);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIFloatSlider::coTUIFloatSlider(const char *n, int pID, bool s)
    : coTUIElement(n, pID, TABLET_FLOAT_SLIDER)
{
    actValue = 0;
    minValue = 0;
    maxValue = 1;
    ticks = 10;

    orientation = s;
    setVal(orientation);
}

coTUIFloatSlider::~coTUIFloatSlider()
{
}

void coTUIFloatSlider::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    tb >> actValue;
    if (i == TABLET_PRESSED)
    {
        if (listener)
        {
            listener->tabletPressEvent(this);
            listener->tabletEvent(this);
        }
    }
    else if (i == TABLET_RELEASED)
    {
        if (listener)
        {
            listener->tabletReleaseEvent(this);
            listener->tabletEvent(this);
        }
    }
    else
    {
        if (listener)
            listener->tabletEvent(this);
    }
}

void coTUIFloatSlider::setValue(float newV)
{
    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void coTUIFloatSlider::setTicks(int newV)
{
    if (ticks != newV)
    {
        ticks = newV;
        setVal(TABLET_NUM_TICKS, ticks);
    }
}

void coTUIFloatSlider::setMin(float minV)
{
    if (minValue != minV)
    {
        minValue = minV;
        setVal(TABLET_MIN, minValue);
    }
}

void coTUIFloatSlider::setMax(float maxV)
{
    if (maxValue != maxV)
    {
        maxValue = maxV;
        setVal(TABLET_MAX, maxValue);
    }
}

void coTUIFloatSlider::setRange(float minV, float maxV)
{
    minValue = minV;
    maxValue = maxV;
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
}

void coTUIFloatSlider::setOrientation(bool o)
{
    orientation = o;
    setVal(orientation);
}

void coTUIFloatSlider::resend()
{
    createSimple(TABLET_FLOAT_SLIDER);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_NUM_TICKS, ticks);
    setVal(actValue);
    setVal(orientation);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUISlider::coTUISlider(const char *n, int pID, bool s)
    : coTUIElement(n, pID, TABLET_SLIDER)
    , actValue(0)
{
    orientation = s;
    setVal(orientation);
}

coTUISlider::~coTUISlider()
{
}

void coTUISlider::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    tb >> actValue;
    if (i == TABLET_PRESSED)
    {
        if (listener)
        {
            listener->tabletPressEvent(this);
            listener->tabletEvent(this);
        }
    }
    else if (i == TABLET_RELEASED)
    {
        if (listener)
        {
            listener->tabletReleaseEvent(this);
            listener->tabletEvent(this);
        }
    }
    else
    {
        if (listener)
            listener->tabletEvent(this);
    }
}

void coTUISlider::setValue(int newV)
{

    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void coTUISlider::setTicks(int newV)
{
    if (ticks != newV)
    {
        ticks = newV;
        setVal(TABLET_NUM_TICKS, ticks);
    }
}

void coTUISlider::setMin(int minV)
{
    minValue = minV;
    setVal(TABLET_MIN, minValue);
}

void coTUISlider::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void coTUISlider::setRange(int minV, int maxV)
{
    minValue = minV;
    maxValue = maxV;
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
}

void coTUISlider::setOrientation(bool o)
{
    orientation = o;
    setVal(orientation);
}

void coTUISlider::resend()
{
    createSimple(TABLET_SLIDER);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_NUM_TICKS, ticks);
    setVal(actValue);
    setVal(orientation);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIComboBox::coTUIComboBox(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_COMBOBOX)
{
    text = NULL;
    selection = -1;
}

coTUIComboBox::~coTUIComboBox()
{
    iter = elements.first();
    while (iter)
    {
        delete[] * iter;
        iter++;
    }
    delete[] text;
}

void coTUIComboBox::parseMessage(TokenBuffer &tb)
{
    delete[] text;
    char *m;
    tb >> m;
    text = new char[strlen(m) + 1];
    strcpy(text, m);
    iter = elements.first();
    int i = 0;
    selection = -1;
    while (iter)
    {
        if (strcmp(*iter, text) == 0)
        {
            selection = i;
            break;
        }
        iter++;
        i++;
    }
    if (listener)
        listener->tabletEvent(this);
}

void coTUIComboBox::addEntry(const char *t)
{
    char *e = new char[strlen(t) + 1];
    strcpy(e, t);
    elements.append(e);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ADD_ENTRY;
    tb << ID;
    tb << e;
    coTabletUI::tUI->send(tb);
}

void coTUIComboBox::delEntry(const char *t)
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_REMOVE_ENTRY;
    tb << ID;
    tb << t;
    coTabletUI::tUI->send(tb);
    iter = elements.first();
    while (iter)
    {
        if (strcmp(*iter, t) == 0)
        {
            iter.remove();
            break;
        }
        iter++;
    }
}

void coTUIComboBox::setSelectedText(const char *t)
{
    delete[] text;
    text = new char[strlen(t) + 1];
    strcpy(text, t);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text;
    coTabletUI::tUI->send(tb);
    int i = 0;
    iter = elements.first();
    while (iter)
    {
        if (strcmp(*iter, text) == 0)
        {
            selection = i;
            break;
        }
        iter++;
        i++;
    }
}

const char *coTUIComboBox::getSelectedText()
{
    return text;
}

int coTUIComboBox::getSelectedEntry()
{
    return selection;
}

void coTUIComboBox::setSelectedEntry(int e)
{
    selection = e;
    if (e >= elements.num())
        selection = elements.num() - 1;
    if (selection < 0)
        return;
    char *selectedEntry = elements.item(selection);
    delete[] text;
    text = new char[strlen(selectedEntry) + 1];
    strcpy(text, selectedEntry);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text;
    coTabletUI::tUI->send(tb);
}

void coTUIComboBox::resend()
{
    createSimple(TABLET_COMBOBOX);
    iter = elements.first();
    while (iter)
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_ADD_ENTRY;
        tb << ID;
        tb << *iter;
        coTabletUI::tUI->send(tb);
        iter++;
    }
    if (text)
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_SELECT_ENTRY;
        tb << ID;
        tb << text;
        coTabletUI::tUI->send(tb);
    }

    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIListBox::coTUIListBox(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_LISTBOX)
{
    text = NULL;
    selection = -1;
}

coTUIListBox::~coTUIListBox()
{
    iter = elements.first();
    while (iter)
    {
        delete[] * iter;
        iter++;
    }
    delete[] text;
}

void coTUIListBox::parseMessage(TokenBuffer &tb)
{
    delete[] text;
    char *m;
    tb >> m;
    text = new char[strlen(m) + 1];
    strcpy(text, m);
    iter = elements.first();
    int i = 0;
    selection = -1;
    while (iter)
    {
        if (strcmp(*iter, text) == 0)
        {
            selection = i;
            break;
        }
        iter++;
        i++;
    }
    if (listener)
        listener->tabletEvent(this);
}

void coTUIListBox::addEntry(const char *t)
{
    char *e = new char[strlen(t) + 1];
    strcpy(e, t);
    elements.append(e);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ADD_ENTRY;
    tb << ID;
    tb << e;
    coTabletUI::tUI->send(tb);
}

void coTUIListBox::delEntry(const char *t)
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_REMOVE_ENTRY;
    tb << ID;
    tb << t;
    coTabletUI::tUI->send(tb);
    iter = elements.first();
    while (iter)
    {
        if (strcmp(*iter, text) == 0)
        {
            iter.remove();
            break;
        }
        iter++;
    }
}

void coTUIListBox::setSelectedText(const char *t)
{
    delete[] text;
    text = new char[strlen(t) + 1];
    strcpy(text, t);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text;
    coTabletUI::tUI->send(tb);
    int i = 0;
    iter = elements.first();
    while (iter)
    {
        if (strcmp(*iter, text) == 0)
        {
            selection = i;
            break;
        }
        iter++;
        i++;
    }
}

const char *coTUIListBox::getSelectedText()
{
    return text;
}

int coTUIListBox::getSelectedEntry()
{
    return selection;
}

void coTUIListBox::setSelectedEntry(int e)
{
    selection = e;
    if (e >= elements.num())
        selection = elements.num() - 1;
    if (selection < 0)
        return;
    char *selectedEntry = elements.item(selection);
    delete[] text;
    text = new char[strlen(selectedEntry) + 1];
    strcpy(text, selectedEntry);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text;
    coTabletUI::tUI->send(tb);
}

void coTUIListBox::resend()
{
    createSimple(TABLET_LISTBOX);
    iter = elements.first();
    while (iter)
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_ADD_ENTRY;
        tb << ID;
        tb << *iter;
        coTabletUI::tUI->send(tb);
        iter++;
    }
    if (text)
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_SELECT_ENTRY;
        tb << ID;
        tb << text;
        coTabletUI::tUI->send(tb);
    }

    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

MapData::MapData(const char *pname, float pox, float poy, float pxSize, float pySize, float pheight)
{
    name = new char[strlen(pname) + 1];
    strcpy(name, pname);
    ox = pox;
    oy = poy;
    xSize = pxSize;
    ySize = pySize;
    height = pheight;
}

MapData::~MapData()
{
    delete[] name;
}

coTUIMap::coTUIMap(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_MAP)
{
}

coTUIMap::~coTUIMap()
{
    iter = maps.first();
    while (iter)
    {
        delete[] * iter;
        iter++;
    }
}

void coTUIMap::parseMessage(TokenBuffer &tb)
{
    tb >> mapNum;
    tb >> xPos;
    tb >> yPos;
    tb >> height;

    if (listener)
        listener->tabletEvent(this);
}

void coTUIMap::addMap(const char *name, float ox, float oy, float xSize, float ySize, float height)
{
    MapData *md = new MapData(name, ox, oy, xSize, ySize, height);
    maps.append(md);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ADD_MAP;
    tb << ID;
    tb << md->name;
    tb << md->ox;
    tb << md->oy;
    tb << md->xSize;
    tb << md->ySize;
    tb << md->height;
    coTabletUI::tUI->send(tb);
}

void coTUIMap::resend()
{
    createSimple(TABLET_MAP);
    iter = maps.first();
    while (iter)
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_ADD_MAP;
        tb << ID;
        tb << iter->name;
        tb << iter->ox;
        tb << iter->oy;
        tb << iter->xSize;
        tb << iter->ySize;
        tb << iter->height;
        coTabletUI::tUI->send(tb);
        iter++;
    }

    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------
//##########################################################

coTUIPopUp::coTUIPopUp(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_POPUP)
{
    text = 0;
    immediate = false;
}

coTUIPopUp::~coTUIPopUp()
{
    delete[] text;
}

void coTUIPopUp::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void coTUIPopUp::parseMessage(TokenBuffer &tb)
{
    char *m;
    tb >> m;
    delete[] text;
    text = new char[strlen(m) + 1];
    strcpy(text, m);
    if (listener)
        listener->tabletEvent(this);
}

void coTUIPopUp::setText(const char *t)
{
    delete[] text;
    if (t)
    {
        text = new char[strlen(t) + 1];
        strcpy(text, t);
        setVal(text);
    }
    else
        text = 0;
}

void coTUIPopUp::resend()
{
    createSimple(TABLET_POPUP);
    if (text)
        setVal(text);
    setVal(immediate);
    coTUIElement::resend();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTabletUI *coTabletUI::tUI = NULL;

//----------------------------------------------------------
//----------------------------------------------------------

coTUIElement::coTUIElement(const char *n, int pID)
{
    xs = -1;
    ys = -1;
    xp = 0;
    yp = 0;
    parentID = pID;
    name = new char[strlen(n) + 1];
    strcpy(name, n);
    label = NULL;
    //	if(coTabletUI::tUI==0)
    //		coTabletUI::tUI = new coTabletUI();
    ID = coTabletUI::tUI->getID();
    coTabletUI::tUI->addElement(this);
    listener = NULL;
}

coTUIElement::coTUIElement(const char *n, int pID, int type)
{
    xs = -1;
    ys = -1;
    xp = 0;
    yp = 0;
    parentID = pID;
    name = new char[strlen(n) + 1];
    strcpy(name, n);
    label = NULL;
    //	if(coTabletUI::tUI==0)
    //		coTabletUI::tUI = new coTabletUI();
    ID = coTabletUI::tUI->getID();
    coTabletUI::tUI->addElement(this);
    listener = NULL;
    createSimple(type);
}

coTUIElement::~coTUIElement()
{
    delete[] name;
    delete[] label;
    TokenBuffer tb;
    tb << TABLET_REMOVE;
    tb << ID;
    coTabletUI::tUI->send(tb);
    coTabletUI::tUI->removeElement(this);
}

void coTUIElement::createSimple(int type)
{
    TokenBuffer tb;
    tb << TABLET_CREATE;
    tb << ID;
    tb << type;
    tb << parentID;
    tb << name;
    coTabletUI::tUI->send(tb);
}

void coTUIElement::setLabel(const char *l)
{
    delete[] label;
    label = new char[strlen(l) + 1];
    strcpy(label, l);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_LABEL;
    tb << ID;
    tb << label;
    coTabletUI::tUI->send(tb);
}

int coTUIElement::getID()
{
    return ID;
}

void coTUIElement::setEventListener(coTUIListener *l)
{
    listener = l;
}

void coTUIElement::parseMessage(TokenBuffer &)
{
}

coTUIListener *coTUIElement::getMenuListener()
{
    return listener;
}

void coTUIElement::setVal(float value)
{
    if (coTabletUI::tUI->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_FLOAT;
    tb << ID;
    tb << value;
    coTabletUI::tUI->send(tb);
}

void coTUIElement::setVal(bool value)
{
    if (coTabletUI::tUI->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_BOOL;
    tb << ID;
    tb << (char)value;
    coTabletUI::tUI->send(tb);
}

void coTUIElement::setVal(int value)
{
    if (coTabletUI::tUI->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_INT;
    tb << ID;
    tb << value;
    coTabletUI::tUI->send(tb);
}

void coTUIElement::setVal(const char *value)
{
    if (coTabletUI::tUI->conn == NULL)
        return;

    cerr << "coTUIElement::setVal info: " << (value ? value : "*NULL*") << endl;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_STRING;
    tb << ID;
    if (value)
        tb << value;
    else
        tb << "";
    coTabletUI::tUI->send(tb);
}

void coTUIElement::setVal(int type, int value)
{
    if (coTabletUI::tUI->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << value;
    coTabletUI::tUI->send(tb);
}

void coTUIElement::setVal(int type, float value)
{
    if (coTabletUI::tUI->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << value;
    coTabletUI::tUI->send(tb);
}
void coTUIElement::setVal(int type, int value, char *nodePath)
{
    if (coTabletUI::tUI->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << value;
    tb << nodePath;
    coTabletUI::tUI->send(tb);
}
void coTUIElement::setVal(int type, char *nodePath, char *simPath, char *simName)
{
    if (coTabletUI::tUI->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << nodePath;
    tb << simPath;
    tb << simName;
    coTabletUI::tUI->send(tb);
}

void coTUIElement::setVal(int type, int value, char *nodePath, char *parentPath)
{
    if (coTabletUI::tUI->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << value;
    tb << nodePath;
    tb << parentPath;
    coTabletUI::tUI->send(tb);
}
void coTUIElement::resend()
{
    TokenBuffer tb;
    if (label)
    {
        tb << TABLET_SET_VALUE;
        tb << TABLET_LABEL;
        tb << ID;
        tb << label;
        coTabletUI::tUI->send(tb);
    }
    if (xs > 0)
    {
        tb.reset();
        tb << TABLET_SET_VALUE;
        tb << TABLET_SIZE;
        tb << ID;
        tb << xs;
        tb << ys;
        coTabletUI::tUI->send(tb);
    }
    tb.reset();
    tb << TABLET_SET_VALUE;
    tb << TABLET_POS;
    tb << ID;
    tb << xp;
    tb << yp;
    coTabletUI::tUI->send(tb);
}

void coTUIElement::setPos(int x, int y)
{
    xp = x;
    yp = y;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_POS;
    tb << ID;
    tb << xp;
    tb << yp;
    coTabletUI::tUI->send(tb);
}

void coTUIElement::setSize(int x, int y)
{
    xs = x;
    ys = y;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SIZE;
    tb << ID;
    tb << xs;
    tb << ys;
    coTabletUI::tUI->send(tb);
}

coTabletUI::coTabletUI()
{
    localHost = NULL;
    serverHost = NULL;
    conn = NULL;
    elements.setNoDelete();
    ID = 3;
    timeout = 0.0;
    tUI = this;
    tryConnect();

    mainFolder = new coTUITabFolder("MainFolder");
    mainFolder->setPos(0, 0);
}

void coTabletUI::close()
{
    delete conn;
    conn = NULL;

    delete serverHost;
    serverHost = NULL;

    delete localHost;
    localHost = NULL;

    elements.setNoDelete();
    ID = 3;
    timeout = 0.0;
    tUI = this;
    tryConnect();
}

void coTabletUI::tryConnect()
{
    delete serverHost;
    serverHost = NULL;
    delete localHost;
    localHost = NULL;
    //const char *line;
    port = 31802;
    //coCoviseConfig::getEntry ("TabletPC.TCPPort", &port);
    //line = coCoviseConfig::getEntry ("TabletPC.Server");

    timeout = 0.0;
    /*coCoviseConfig::getEntry("TabletPC.Timeout", &timeout);
   if (line)
   {
      if(strcasecmp(line,"NONE") != 0)
      {
         serverHost = new Host(line);
         localHost = new Host("localhost");
      }
   }
   else
   {
      localHost = new Host("localhost");
   }*/
    serverHost = new Host("localhost");
    localHost = new Host("localhost");
}

coTabletUI::~coTabletUI()
{
    delete mainFolder;
    delete conn;
    delete serverHost;
    delete localHost;
}

int coTabletUI::getID()
{
    return ID++;
}

void coTabletUI::send(TokenBuffer &tb)
{
    if (conn == NULL)
        return;
    Message m(tb);
    m.type = TABLET_UI;
    conn->send_msg(&m);
}

VOID CALLBACK coTabletUI::timerCallback(HWND hwnd,
                                        UINT uMsg,
                                        UINT_PTR idEvent,
                                        DWORD dwTime)
{
    if (coTabletUI::tUI)
        coTabletUI::tUI->update();
}

void coTabletUI::update()
{
    /*if(coVRMSController::msController==NULL)
      return;*/

    static double oldTime = 0.0;
    if (conn)
    {
    }
    else if (/*(coVRMSController::msController->isMaster()) && */ (serverHost != NULL || localHost != NULL))
    {
        // try to connect to server every 2 secnods
        //if((cover->frameTime() - oldTime)>2)
        {
            if (serverHost)
            {
                conn = new ClientConnection(serverHost, port, 0, (sender_type)0, 0, timeout);
            }
            if (conn && !conn->is_connected()) // could not open server port
            {
#ifndef _WIN32
                if (errno != ECONNREFUSED)
                {
                    fprintf(stderr, "Could not connect to TabletPC %s; port %d: %s\n",
                            serverHost->get_name(), port, strerror(errno));
                    delete serverHost;
                    serverHost = NULL;
                }
#else
                fprintf(stderr, "Could not connect to TabletPC %s; port %d\n", serverHost->get_name(), port);
                delete serverHost;
                serverHost = NULL;
#endif
                delete conn;
                conn = NULL;
            }
            if (conn && conn->is_connected())
            {
                // create Texture and SGBrowser Connections
                Message *msg = new Message();
                conn->recv_msg(msg);
                TokenBuffer tb(msg);
                int tPort;
                tb >> tPort;

                ClientConnection *cconn = new ClientConnection(serverHost, tPort, 0, (sender_type)0, 0);
                if (!cconn->is_connected()) // could not open server port
                {
#ifndef _WIN32
                    if (errno != ECONNREFUSED)
                    {
                        fprintf(stderr, "Could not connect to TabletPC TexturePort %s; port %d: %s\n",
                                connectedHost->getName(), tPort, strerror(errno));
                    }
#else
                    fprintf(stderr, "Could not connect to TabletPC; port %d\n", tPort);
#endif
                    delete cconn;
                    cconn = NULL;
                }
                textureConn = cconn;

                conn->recv_msg(msg);
                TokenBuffer stb(msg);

                stb >> tPort;

                cconn = new ClientConnection(serverHost, tPort, 0, (sender_type)0, 0);
                if (!cconn->is_connected()) // could not open server port
                {
#ifndef _WIN32
                    if (errno != ECONNREFUSED)
                    {
                        fprintf(stderr, "Could not connect to TabletPC TexturePort %s; port %d: %s\n",
                                connectedHost->getName(), tPort, strerror(errno));
                    }
#else
                    fprintf(stderr, "Could not connect to TabletPC localhost; port %d\n", tPort);
#endif
                    delete cconn;
                    cconn = NULL;
                }

                sgConn = cconn;
            }

            if (!conn && localHost)
            {
                conn = new ClientConnection(localHost, port, 0, (sender_type)0, 0);
                if (!conn->is_connected()) // could not open server port
                {
#ifndef _WIN32
                    if (errno != ECONNREFUSED)
                    {
                        fprintf(stderr, "Could not connect to TabletPC %s; port %d: %s\n",
                                localHost->get_name(), port, strerror(errno));
                        delete localHost;
                        localHost = NULL;
                    }
#else
                    fprintf(stderr, "Could not connect to TabletPC %s; port %d\n", localHost->get_name(), port);
                    delete localHost;
                    localHost = NULL;
#endif
                    delete conn;
                    conn = NULL;
                }
            }

            if (conn && conn->is_connected())
            {
                // resend all ui Elements to the TabletPC
                coDLListIter<coTUIElement *> iter;
                iter = elements.first();
                while (iter)
                {
                    iter->resend();
                    iter++;
                }
            }
            else
            {
                // if(VRCoviseConnection::covconn)
                // {
                //    CoviseRender::send_ui_message("WANT_TABLETUI", "");
                // }
            }
            //oldTime = cover->frameTime();
        }
    }

    bool gotMessage = false;
    do
    {
        gotMessage = false;
        Message m;
        /*if(coVRMSController::msController->isMaster())*/
        {
            if (conn)
            {
                if (conn->check_for_input())
                {
                    conn->recv_msg(&m);
                    gotMessage = true;
                }
            }
            /*coVRMSController::msController->sendSlaves((char *)&gotMessage,sizeof(bool));
         if(gotMessage)
         {
            coVRMSController::msController->sendSlaves(&m);
         }*/
        }
        /* else
      {
         if(coVRMSController::msController->readMaster((char *)&gotMessage,sizeof(bool))<0)
         {
            cerr << "bcould not read message from Master" << endl;
            exit(0);
         }
         if(gotMessage)
         {
            if(coVRMSController::msController->readMaster(&m)<0)
            {
               cerr << "ccould not read message from Master" << endl;
               //cerr << "sync_exit13 " << myID << endl;
               exit(0);
            }
         }
      }*/
        if (gotMessage)
        {

            TokenBuffer tb(&m);
            switch (m.type)
            {
            case SOCKET_CLOSED:
            case CLOSE_SOCKET:
            {
                delete conn;
                conn = NULL;
            }
            break;
            case TABLET_UI:
            {

                int ID;
                tb >> ID;
                if (ID >= 0)
                {
                    //coDLListSafeIter<coTUIElement*> iter;
                    coDLListIter<coTUIElement *> iter;
                    iter = elements.first();
                    while (iter)
                    {
                        if (*iter)
                        {
                            if (iter->getID() == ID)
                            {
                                iter->parseMessage(tb);
                                break;
                            }
                        }
                        iter++;
                    }
                }
            }
            break;
            default:
            {
                cerr << "unknown Message type" << endl;
            }
            break;
            }
        }
    } while (gotMessage);
}

void coTabletUI::addElement(coTUIElement *e)
{
    elements.append(e);
}

void coTabletUI::removeElement(coTUIElement *e)
{
    coDLListIter<coTUIElement *> iter;
    iter = elements.findElem(e);
    if (iter)
        iter.remove();
}

void coTUIListener::tabletEvent(coTUIElement *)
{
}

void coTUIListener::tabletPressEvent(coTUIElement *)
{
}

void coTUIListener::tabletReleaseEvent(coTUIElement *)
{
}
