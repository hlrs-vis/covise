import java.awt.*;
import java.applet.*;
import java.awt.image.*;
import java.awt.event.*;
import javax.swing.*;
import java.util.*;
import java.lang.*;
import java.net.*;
import java.io.*;

import netscape.javascript.*;

import com.lotus.sametime.core.comparch.*;
import com.lotus.sametime.core.constants.*;
import com.lotus.sametime.core.types.*;
import com.lotus.sametime.community.*;
import com.lotus.sametime.places.*;
import com.lotus.sametime.awarenessui.*;
import com.lotus.sametime.awarenessui.placelist.*;
import com.lotus.sametime.conference.*;
import com.lotus.sametime.conference.event.*;
import com.lotus.sametime.streamedmedia.*;
import com.lotus.sametime.applet.STApplet;
import com.lotus.sametime.appshare.*;
import com.lotus.sametime.appshare.event.*;
import com.lotus.sametime.whiteboard.*;
import com.lotus.sametime.whiteboard.event.*;

/** 
 * Meeting Applet sample showing how to create a meeting with Video
 */
public class MeetingLauncher extends STApplet
{
  private STSession m_session;		  
  private CommunityService m_comm;
  private Place m_place;
  private PlaceAwarenessList m_peopleList;
    
  protected static Panel mainPanel;
  private Panel mainLeftPanel;
  private Panel mainRightPanel;
  private Panel mainWestPanel;
  private Panel mainEastPanel;
  private Panel mainSharePanel;
  private Panel mainShareControlPanel;
  private Panel mainWBPanel;
  private Panel mainWBControlPanel;
  private Panel barreoutilsWBControlPanel;
  private Panel mainVideoPanel;
  private Panel centerPanel;
  private Panel eastPnl;
  private Panel m_audioPanel;
  private Panel m_videoPanel;
  private AppShare m_appshare;
  private Panel listAppliPanel;
  private Panel listAppliPanelContent;
  private Panel listAppliPanelShape;
  private Dialog QuitBox;

  public static Frame mainFrame;
  
  private AudioManager m_audioManager;
  protected static VideoManager m_videoManager;
  private AppShareControls AppShare_Host;
  private WhiteboardService whiteboardService;
  private Label m_PeopleNumLbl;

  private Dimension dimensionEcran;
  private AppShareComp m_appshareservice;
  private Frame permModerator;
  	
  private static int numUsersInPlace = 0;
  private STUser[] users;
  private STUser user;
  private STUser actualUser;
  private STExtendedAttribute moderatorName;
  private STExtendedAttribute placemoderatorName;
  private STExtendedAttribute moderationName;
  private URLConnection urlConnection;
  private TextArea codeDisplay;
  private String componentName;
  //private String searchString;
  //static boolean windowClosed;
  
  private PlaceAwarenessList place_List;
  private MyselfInPlace myself_in_Place;
  private Activity Meeting_Activity;
  public static final int DATA_PERMISSION =1;
  public static final int AV_PERMISSION =2;
  public static final int MEETING_MODERATOR =3;
  
  public static final int GRANT_AV_PERMISSION =1;
  public static final int REVOKE_AV_PERMISSION =2;
  public static final int GRANT_DATA_PERMISSION =3;
  public static final int REVOKE_DATA_PERMISSION =4;
  public static final int GRANT_AV_TO_ALL =5;
  public static final int REVOKE_AV_FROM_ALL =6;
  public static final int GRANT_DATA_TO_ALL =7;
  public static final int REVOKE_DATA_FROM_ALL =8;
  public static final int SWITCH_MODERATOR =9;
  protected Button Share_Launcher_Button;
  private String STARTSHARE ="Start Application Sharing";
  private String STOPSHARE ="Stop Application Sharing";
  private Button WB_Launcher_Button;
  private String STARTWB =   "Start White Board        ";
  private String STOPWB = STARTSHARE;
  private Button AVideo_Launcher_Button;
  
  public static int maxX = 1550;
  public static int maxY = 10000;
  
  private MotifButton whiteBoardButton=null;
  public static boolean whiteboard_is_active;

  static Image logo;
  Image motifButton;
  static Image quitIcon;
  Image questionIcon, blankImage;
  Image startVideoIcon;
  Image stopVideoIcon;
  Image startWhiteboardIcon;
  static Image startAppSharingIcon;
  static Image stopAppSharingIcon;
  static Image box;
  static Image box_checked;

  public static Image bnorth, bsouth, bwest, beast;

  private Color bg_color;
  private Color border_color;

  /** 
   * Initialize the applet. Create the session, load and start
   * the components.
   */
  public void init()
  {
    bg_color = new Color(0xdedede);
    border_color = new Color(0xaaaaaa);

    try
    {
      m_session = new STSession("MeetingApplet " + this);
    }
    catch (DuplicateObjectException	exception)
    {
      exception.printStackTrace();
    }

    m_session.loadAllComponents();
		
    try
    {
      new MeetingFactoryComp(
            m_session,
            getCodeBase().toString(),
            getParameter("MeetingResourceNamespace"),
            this
            );
      StreamedMediaService streamedMediaService = new StreamedMediaComp(m_session);
      whiteboardService = new WhiteboardComp(m_session);
      m_appshareservice = new AppShareComp(m_session);
    }
    catch(DuplicateObjectException e)
    {
      e.printStackTrace();
    }
    		
    m_session.start();
        
    m_comm = (CommunityService)m_session.getCompApi(CommunityService.COMP_NAME);
    m_comm.addLoginListener(new CommunityEventsListener());
    
    String community = getCodeBase().getHost().toString();
    String loginName = getParameter("loginName");
    String password = getParameter("password");		
    //m_comm.loginByPassword(community, loginName, password); 
    m_comm.loginAsAnon(community, loginName);
  }
	
  /** 
   * Enter the place the meeting will take place in.
   */
  public void enterPlace()
  {
    PlacesService placesService = (PlacesService)m_session.getCompApi(PlacesService.COMP_NAME);
    
    m_place = placesService.createPlace(	
                    getParameter("placeName"), // place unique name 
                    getParameter("placeName"), // place display name 
                    EncLevel.ENC_LEVEL_DONT_CARE, // encryption level
                    0,                            // place type
                    PlacesConstants.PLACE_PUBLISH_DONT_CARE);
    
    m_place.addPlaceListener(new PlaceEventsListener());
    m_place.enter();
  }
	
  /** 
   * Layout the applet UI.
   */
  protected void layoutAppletUI()
  {
    whiteboard_is_active = true;

    /// get screen size
    
    Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
    if( screenSize.width < maxX ) maxX = screenSize.width;
    if( screenSize.height < maxY ) maxY = screenSize.height;    

    mainFrame = new Frame();
    maximize_UI();
    //mainFrame.setSize( maxX,maxY);
    //mainFrame.setLocation(1,-25);
    //mainFrame.setTitle("COVISE Conference Room");

    loadImages();        

    layoutWestPanel();
    layoutEastPanel();
    layoutListAppliPanel();
    
    /*Label infoLabel = new Label("All data transfer are encrypted. ");
    infoLabel.setFont(new Font("Dialog",Font.PLAIN,18));
    infoLabel.setAlignment(infoLabel.CENTER);
    */
    centerPanel = new Panel(new BorderLayout());
    centerPanel.setBackground(bg_color);
    
    mainLeftPanel = new Panel(new BorderLayout());
    mainRightPanel = new Panel(new BorderLayout());
    mainRightPanel.setBackground(bg_color);
    //mainRightPanel.add(infoLabel,BorderLayout.NORTH);
    mainRightPanel.add(centerPanel,BorderLayout.CENTER);
    //mainRightPanel.add( mainLeftPanel, BorderLayout.CENTER );
    //mainRightPanel.add(mainWestPanel,BorderLayout.WEST);
    mainRightPanel.add(mainEastPanel,BorderLayout.EAST);
    mainRightPanel.add(listAppliPanel,BorderLayout.SOUTH);            


    mainPanel = new Panel(new GridLayout(1,1));
    //mainPanel.add(mainLeftPanel);
    mainPanel.add(mainRightPanel);
    mainFrame.add("Center",mainPanel);
    mainFrame.addWindowListener(new EcouteurQuitterAppli());    
    mainPanel.validate();
    layoutQuitBox();
   
    mainFrame.show();    
  }

  /**
  *  Put the UI to maximum screen size
  */
  public static void maximize_UI() {
    mainFrame.setSize(maxX, maxY);
    mainFrame.setLocation(0,0);
    mainFrame.setTitle("COVISE Conference Room");
  }

  /**
  *  Put the UI to the size of the Application Sharing Control panel
  */
  public static void minimize_UI() {
    mainFrame.setSize(maxX,100);
    mainFrame.setLocation(0, maxY-100);
    mainFrame.setTitle("COVISE Conference Room - Application Sharing Controls");
  }
  
  /**
  *  Function returns true if Whiteboard is active and visible in the applet. Returning of false means: applet is in
  *  application sharing mode
  */
  public static boolean isWhiteboardActive() {
    return( whiteboard_is_active );
  }
 
 
  /**
  * Load all images
  * This has to be done in the applet class and therefore central 
  */  
  private void loadImages() {
     try {        
       logo = getImage( new URL( getCodeBase(), "wuerfel.gif") );
       
       motifButton = getImage( new URL( getCodeBase(), "motifButton2.gif") );       
       
       quitIcon = getImage( new URL( getCodeBase(), "quit.gif") );
              
       startVideoIcon = getImage( new URL( getCodeBase(), "video.gif") );                    
       stopVideoIcon = getImage( new URL( getCodeBase(), "novideo.gif") );
       
       startWhiteboardIcon = getImage( new URL( getCodeBase(), "pen.gif") );              
       startAppSharingIcon = getImage( new URL( getCodeBase(), "appl.gif") );              
       stopAppSharingIcon = getImage( new URL( getCodeBase(), "noappl.gif") );
      
       box = getImage( new URL( getCodeBase(), "box.gif") );             
       box_checked = getImage( new URL( getCodeBase(), "box-checked.gif") );       
                       
       bwest = getImage( new URL( getCodeBase(), "west.gif") );
       beast = getImage( new URL( getCodeBase(), "east.gif") );
       bnorth = getImage( new URL( getCodeBase(), "north.gif") );
       bsouth = getImage( new URL( getCodeBase(), "south.gif") );
       
       questionIcon = getImage( new URL( getCodeBase(), "quest.gif") );

       mainFrame.setIconImage(logo);
    }
    catch ( MalformedURLException mue ) { };
  }
  
   /** 
   * Layout the East Panel.
   */
  protected void layoutEastPanel()
  {
    mainVideoPanel = new Panel(new BorderLayout());
    mainVideoPanel.setBackground(bg_color);
    Panel NPanel = new Panel(new BorderLayout());
    Panel peopleHerePnl = new Panel(new FlowLayout(FlowLayout.LEFT));
    eastPnl = new Panel(new BorderLayout());
    eastPnl.setBackground(bg_color);
    mainEastPanel = new Panel(new BorderLayout());
    Label PeopleHereLbl = new Label("People Here:");
    PeopleHereLbl.setFont(new Font("Dialog",Font.BOLD,14));
    peopleHerePnl.add(PeopleHereLbl);
    m_PeopleNumLbl = new Label("0");
    m_PeopleNumLbl.setFont(new Font("Dialog",Font.PLAIN,14));
    peopleHerePnl.add(m_PeopleNumLbl);
    NPanel.add(peopleHerePnl, BorderLayout.NORTH);
    m_peopleList = new PlaceAwarenessList(m_session, true);      													
    m_peopleList.addAwarenessViewListener(new ParticipantListListener());
    eastPnl.add(NPanel,BorderLayout.NORTH);
    eastPnl.add(m_peopleList,BorderLayout.CENTER);
    eastPnl.validate();
    mainEastPanel.add(eastPnl,BorderLayout.CENTER);
  }
  
   /** 
   * Layout the West Panel.
   */
  protected void layoutWestPanel()
  {
    mainWestPanel = new Panel(); //new GridLayout(6,1));
    mainWestPanel.setBackground(bg_color);
    mainWBPanel = new Panel(new BorderLayout());
    mainSharePanel = new Panel(new BorderLayout());
    mainShareControlPanel = new Panel();
    mainShareControlPanel.setBackground(bg_color);
   
    mainWBControlPanel = new Panel(new GridLayout(2,1));
    mainWBControlPanel.setBackground(bg_color);
    barreoutilsWBControlPanel = new Panel(new GridLayout(2,1));
    barreoutilsWBControlPanel.setBackground(bg_color);
  }
  
  /** 
   * Layout the Application List Interface.
   */
  protected void layoutListAppliPanel()
  {        
    listAppliPanelContent = new Panel();
    listAppliPanelContent.setLayout(new BorderLayout( 10, 0) );   
   
    Panel blank_top = new Panel();
    listAppliPanelContent.add(blank_top,BorderLayout.NORTH);     
    
    Panel blank_bottom = new Panel();
    listAppliPanelContent.add(blank_bottom,BorderLayout.SOUTH);
            
    //listAppliPanelContent.add( new AppWhiteboardButton() );
    Share_Launcher_Button = new Button(STARTSHARE);
    Share_Launcher_Button.addActionListener(new ShareLauncherButtonHandler());
    WB_Launcher_Button = new Button(STARTWB);
    WB_Launcher_Button.addActionListener(new WBLauncherButtonHandler());
    
    whiteBoardButton = new MotifButton(startWhiteboardIcon, 
                                               startAppSharingIcon,
					       "    Whiteboard", "Application Sharing",  190 );
    WhiteboardListener wl = new WhiteboardListener();
    wl.setButton(whiteBoardButton);
    whiteBoardButton.addMouseListener(wl);
    whiteBoardButton.addMouseMotionListener(wl);
    listAppliPanelContent.add( whiteBoardButton, BorderLayout.WEST);

    MotifButton video = new MotifButton(startVideoIcon, stopVideoIcon, "Start Video", "Stop Video", 125 );
    VideoListener vl = new VideoListener();
    vl.setButton(video);
    video.addMouseListener(vl);
    video.addMouseMotionListener(vl);
    listAppliPanelContent.add( video, BorderLayout.CENTER);
 
    MotifButton qu = new MotifButton(quitIcon, quitIcon, "Quit", "Quit", 80 );
    QuitListener ql = new QuitListener();
    qu.addMouseListener(ql);
    qu.addMouseMotionListener(ql);
    listAppliPanelContent.add( qu, BorderLayout.EAST);
   
    Button aBoutton = new Button("");
    Button anotherBoutton = new Button("");
    Panel aPanel = new Panel(new GridLayout(2,1));
    Panel anotherPanel = new Panel(new GridLayout(2,1));
    anotherPanel.add(anotherBoutton);
    aPanel.add(aBoutton);
    
    //listAppliPanelContent.add(aPanel);
    aPanel.setVisible(false);         
    
    //listAppliPanelContent.add(anotherPanel);
    anotherPanel.setVisible(false);

    listAppliPanelShape = new Panel( new BorderLayout());
    Button blank = new Button("Control Panel");
    blank.setFont(new Font("Dialog",Font.BOLD,8));
    blank.setBackground( bg_color);
    blank.setSize(40,3);
    blank.setEnabled(false);


    //listAppliPanelShape.add( blank, BorderLayout.NORTH);
    //listAppliPanelShape.add( blankPanel, BorderLayout.NORTH);
    //listAppliPanelShape.add( listAppliPanelContent, BorderLayout.CENTER);    

    listAppliPanel = new Panel( new BorderLayout() );
    listAppliPanel.add( blank, BorderLayout.NORTH);
    Panel p = new Panel( new GridLayout(1,2) );
    p.add(mainShareControlPanel);
    p.add(listAppliPanelContent);
    listAppliPanel.add(p, BorderLayout.SOUTH);
    //listAppliPanel.add( mainShareControlPanel, BorderLayout.WEST);    
    //listAppliPanel.add( listAppliPanelContent, BorderLayout.EAST);
        
    listAppliPanel.validate();
  }
  

  class QuitListener extends MouseAdapter 
		     implements MouseMotionListener {
        public void mousePressed(MouseEvent e) {
	   QuitBox.setVisible(true); 
	}

	public void mouseDragged(MouseEvent e) { }

	public void mouseMoved(MouseEvent e) {}

        public void mouseReleased(MouseEvent e) {}
   }
    

  class VideoListener extends MouseAdapter 
		  implements MouseMotionListener {
     //
     // button that corresponds to the Listener	  
     //
     MotifButton _videoButton; 
     
     /// set corresponding button
     public void setButton( MotifButton videoButton) {
      _videoButton = videoButton;
     } 

     public void mousePressed(MouseEvent e) {	              
	if (_videoButton.getState())
	{
             _videoButton.setState(false);
             System.out.println("action performed audiovideo");
             mainEastPanel.add( encapsulate(mainVideoPanel, border_color, false, false, true, false) ,BorderLayout.NORTH);
             //mainEastPanel.add( mainVideoPanel ,BorderLayout.NORTH);
             mainEastPanel.add(eastPnl);
             mainPanel.validate();
             //m_audioManager.startAudioDevices();
             m_videoManager.startVideoDevices();                                
             System.out.println("end action performed audiovideo");
	}
	else
	{
            _videoButton.setState(true);
            //m_audioManager.stopAudioDevices();
            m_videoManager.stopVideoDevices();               
            mainEastPanel.removeAll();
            mainPanel.validate();               
	}
	_videoButton.repaint();	
     }

     public void mouseDragged(MouseEvent e) {}

     public void mouseMoved(MouseEvent e) {}

     public void mouseReleased(MouseEvent e) {}
    }

   class WhiteboardListener extends MouseAdapter 
		     implements MouseMotionListener {
        MotifButton _whiteboardButton; 
        public void setButton( MotifButton whiteboardButton) {
         _whiteboardButton = whiteboardButton;
        } 

        public void mousePressed(MouseEvent e) {	              
           if ( _whiteboardButton.getState()) {           
                _whiteboardButton.setState(false);
                System.out.println("action performed white board");
                centerPanel.removeAll();
                Panel basicWBPanel=new Panel(new BorderLayout());
                basicWBPanel.setBackground(new Color(0xffffff) );
                basicWBPanel.add(mainWBPanel,BorderLayout.CENTER);
                basicWBPanel.add(mainWBControlPanel,BorderLayout.NORTH);
                centerPanel.add(encapsulate(basicWBPanel, border_color,false, true,false, true),BorderLayout.CENTER);
                
                //mainLeftPanel.removeAll();
                mainShareControlPanel.setVisible(false);
                //mainWestPanel.removeAll();
                //AppShare_Host.shutdown();

                mainRightPanel.remove( (Component) mainLeftPanel);
                mainRightPanel.add( centerPanel, BorderLayout.CENTER);

                mainPanel.validate();
                //mainFrame.toFront();
                
                maximize_UI();
                
                mainFrame.validate();
                
                whiteboard_is_active = true;
                System.out.println("end action performed white board");
           }
           else {          
               _whiteboardButton.setState(true);
               //mainLeftPanel.add(mainSharePanel,BorderLayout.CENTER);
               mainShareControlPanel.setVisible(true);
               //mainWestPanel.add(mainShareControlPanel,BorderLayout.NORTH);
               mainPanel.validate();

               centerPanel.removeAll();
               centerPanel.setBackground(bg_color);
               mainRightPanel.remove( (Component) centerPanel);
               mainRightPanel.add( mainLeftPanel, BorderLayout.CENTER);
               mainPanel.validate();
                 
               //if( (AppShareControls.Share_Window_Button.isVisible() &&
               //      !AppShareControls.Allow_Drive_Checkbox.isVisible()) || AppShareControls.Allow_Drive_Checkbox.isVisible() ) {  
               if( AppShareControls.noApplicationShared() ) {
        	 minimize_UI(); 
                 mainFrame.validate(); 
               }      
               whiteboard_is_active = false;                  
           }
           _whiteboardButton.repaint();
	}

	public void mouseDragged(MouseEvent e) {}

	public void mouseMoved(MouseEvent e) {}

        public void mouseReleased(MouseEvent e) {}
   }  

  /** 
  * This method is called when the user clicks on the "YES" Button in the QuitBox
  */
  class QuitAppletHandler implements ActionListener { 
 
       /**
        * Called if user confirms in quit box
        *
        */
       
       public void actionPerformed(ActionEvent	event)
        {
           System.out.println("action performed exit");
           stop();
           destroy();
           System.out.println("end action performed exit");
        }
  }
 
  /** 
  * This method is called when the user clicks on the "NO" Button in the QuitBox
  */
  class DontQuitAppletHandler implements ActionListener
  {
       /**
        * Called if user does not confirm in quit box
        *
        */
       public void actionPerformed(ActionEvent	event)
        {
           QuitBox.setVisible(false);
        }
  }

  /**
  * Layout for the QuitBox: comfirmation dialog
  */

  protected void layoutQuitBox()
  {          
     Button Yes_Button = new Button(" Yes ");
     Yes_Button.setFont( new Font("Dialog", Font.PLAIN,14 ));
     Yes_Button.addActionListener(new QuitAppletHandler());    
     Yes_Button.setBackground( bg_color);

     Button No_Button  = new Button(" No ");
     No_Button.setFont( new Font("Dialog", Font.BOLD,14 ));
     No_Button.addActionListener(new DontQuitAppletHandler());     
     No_Button.setBackground( bg_color);

     Panel QuitBoxButtonPanel = new Panel( new FlowLayout(FlowLayout.CENTER, 5, 10) );
     QuitBoxButtonPanel.add( Yes_Button);
     QuitBoxButtonPanel.add( No_Button);                                       
          
     MotifButton QuitQuestion = new MotifButton( questionIcon, questionIcon, 
                                                        "   Are you sure to quit?", "   Are you sure to quit?", 150);     
     QuitQuestion.removeBorders();
     Panel QuitQuestionPanel = new Panel( new GridLayout(3,1) );
     QuitQuestionPanel.add(new Label(""));
     QuitQuestionPanel.add(QuitQuestion);     
     QuitQuestionPanel.add(new Label(""));

     Panel QuitBoxPanel = new Panel( new BorderLayout() );
     QuitBoxPanel.add( QuitQuestionPanel, BorderLayout.CENTER);
     QuitBoxPanel.add( QuitBoxButtonPanel, BorderLayout.SOUTH);
     
     QuitBox = new Dialog( mainFrame);
     QuitBox.setTitle("Quit COVISE Conference Room");
     QuitBox.setLocation(400, 400);
     QuitBox.setBackground(bg_color);
     QuitBox.setSize(210,190);
     QuitBox.add( QuitBoxPanel);     
     //QuitBox.pack();
     QuitBox.setVisible( false);
     QuitBox.validate();     
  }

   /** 
   * A method to add border(s) to a panel.
   */
  
  public Panel encapsulate(Panel srcPanel,Color capColor,boolean north, boolean east, boolean south, boolean west)
  {
     Panel northPanel = new Panel();
     northPanel.setBackground(capColor);     
     Panel eastPanel = new Panel();
     eastPanel.setBackground(capColor);    
     Panel southPanel = new Panel();
     southPanel.setBackground(capColor);
     Panel westPanel = new Panel();
     westPanel.setBackground(capColor);
     Panel destPanel = new Panel(new BorderLayout());
     destPanel.add(srcPanel,BorderLayout.CENTER);
     if (north) destPanel.add(northPanel,BorderLayout.NORTH);
     if (east) destPanel.add( eastPanel,BorderLayout.EAST);
     if (south) destPanel.add(southPanel,BorderLayout.SOUTH);
     if (west) destPanel.add(westPanel,BorderLayout.WEST);
     return destPanel;
  }  

  /** 
   * A listener for loggedIn/loggedOut events.
   */
  class CommunityEventsListener implements LoginListener
  {
    public void loggedIn(LoginEvent event)
    {
      layoutAppletUI();
      enterPlace();
    }

    public void loggedOut(LoginEvent event)
    {
    }	
  }
  
  class ShareLauncherButtonHandler implements ActionListener
  {
	/**
	 * Called when the Share Launcher button has been 
	 * pressed.  A new Panel is launched containing the shared application
	 * 
	 * @param event
	 *  The ActionEvent received from the Share Launcher button.
         */
	public void actionPerformed(ActionEvent	event)
        {
           
        }
  }
  
 
  class WBLauncherButtonHandler implements ActionListener
  {
	/**
	 * Called when the White Board Launcher button has been 
	 * pressed.  A new Panel is launched containing the wb application
	 * 
	 * @param event
	 *  The ActionEvent received from the WB Launcher button.
         */
	public void actionPerformed(ActionEvent	event)
        {
           
        }
  }

  
  /** 
   * A listener for appshare events.
   */
  class AppShareHandler extends AppShareAdapter
  {
    
      public void appshareAvailable(AppShareEvent event)
      {
          System.out.println("begin appshareavailable");
          AppShare_Host = new AppShareControls(m_appshare);
          Label asLabel = new Label("Application Sharing Controls");
          asLabel.setFont(new Font("Dialog",Font.PLAIN,16));
          asLabel.setAlignment(asLabel.CENTER);
          asLabel.setBackground(bg_color);
          //mainShareControlPanel.add(asLabel,BorderLayout.NORTH);
          mainShareControlPanel.add(AppShare_Host.getViewableComponent());//,BorderLayout.CENTER);
	  mainSharePanel.add(m_appshare.getViewableComponent(true),BorderLayout.CENTER);
          mainPanel.validate();
          System.out.println("end appshareavailable");
      }

  }

   /** 
   * A listener for launcher events.
   */
    public class EcouteurQuitterAppli extends WindowAdapter
    {
    
        public void windowClosing(WindowEvent we)
        {
            //stop();
            //destroy();
            //System.exit(0);
            QuitBox.setVisible(true);
        }
    }  
    
    public class EcouteurQuitterFrame extends WindowAdapter
    {
    
        public void windowClosing(WindowEvent we)
        {
            we.getWindow().removeAll();
            we.getWindow().dispose();
        }
    }  

    
  /** 
   * A listener for place events.
   */
  class PlaceEventsListener extends PlaceAdapter
  {
    public void entered(PlaceEvent event)
    {   
      myself_in_Place = event.getPlace().getMyselfInPlace();
      myself_in_Place.addUserInPlaceListener(new UserHandler());
      
      m_peopleList.bindToSection(event.getPlace().getMySection());      			    			
      event.getPlace().addActivity(StreamedMedia.AUDIO_ACTIVITY_TYPE, null);
      event.getPlace().addActivity(StreamedMedia.VIDEO_ACTIVITY_TYPE, null);
      event.getPlace().addActivity(Whiteboard.WHITEBOARD_ACTIVITY_TYPE, null);
      event.getPlace().addActivity(AppShare.APPSHARE_ACTIVITY_TYPE, null);
      
      //someone MUST be the moderator. Make it yourself.
      
      //moderatorName = new STExtendedAttribute(3,true);
      //myself_in_Place.changeAttribute(moderatorName);
     
      place_List = new PlaceAwarenessList(m_session);
      
      System.out.println("get attributes : " + event.getPlace().getAttributes()); 
      for (Enumeration e = event.getPlace().getAttributes() ; e.hasMoreElements() ;)
      {
         System.out.println(e.nextElement());
      }
      System.out.println("member id : " + myself_in_Place.getMemberId() + "  |  login name :" + MeetingLauncher.this.getParameter("loginName") + "  |  meeting moderator :" + MeetingFactoryService.MEETING_MODERATOR);
      //placemoderatorName = new STExtendedAttribute(MeetingFactoryService.MEETING_MODERATOR,MeetingLauncher.this.getParameter("loginName"));
      //placemoderatorName = new STExtendedAttribute(3,user.getId().getId().getBytes());
      //event.getPlace().changeAttribute(placemoderatorName);
      System.out.println("get attributes : " + event.getPlace().getAttributes()); 
      for (Enumeration e = event.getPlace().getAttributes() ; e.hasMoreElements() ;)
      {
         System.out.println("yoyo "+(String)e.nextElement());
      }
    }
    
    public void activityAdded(PlaceEvent event)
    {
      Meeting_Activity = event.getActivity();
      if (event.getActivityType() == AppShare.APPSHARE_ACTIVITY_TYPE)
      {
        //System.out.println("add appshare");
        m_appshare = m_appshareservice.getAppShareForPlace(m_place);
        m_appshare.addAppShareListener(new AppShareHandler());
        mainLeftPanel.add(mainSharePanel,BorderLayout.CENTER);
        //mainWestPanel.add(mainShareControlPanel,BorderLayout.NORTH);
        mainShareControlPanel.setVisible(false);
        //Share_Launcher_Button.setLabel(STOPSHARE);
        //System.out.println("fin add appshare");
      }
      if (event.getActivityType() ==  Whiteboard.WHITEBOARD_ACTIVITY_TYPE)
      {
        System.out.println("add wb");
        Whiteboard m_whiteboard = whiteboardService.getWhiteboardForPlace(m_place);
        Label wbLabel = new Label("White Board Controls");
        wbLabel.setFont(new Font("Dialog",Font.BOLD,16));
        wbLabel.setAlignment(wbLabel.CENTER);
        wbLabel.setBackground(bg_color);
        barreoutilsWBControlPanel.add(wbLabel);
        barreoutilsWBControlPanel.add(m_whiteboard.getDefaultToolbar());
        mainWBControlPanel.add(barreoutilsWBControlPanel);
        mainWBControlPanel.add(m_whiteboard.getDefaultControlPanel());
        mainWBPanel.add(m_whiteboard.getViewableComponent(false),BorderLayout.CENTER);
        centerPanel.removeAll();
        Panel basicWBPanel=new Panel(new BorderLayout());
        basicWBPanel.setBackground(new Color(0xffffff));
        basicWBPanel.add(mainWBPanel,BorderLayout.CENTER);
        basicWBPanel.add(mainWBControlPanel,BorderLayout.NORTH);
        centerPanel.add(encapsulate(basicWBPanel, border_color,false,true,true,true),BorderLayout.CENTER);
        WB_Launcher_Button.setLabel(STOPWB);
        System.out.println("fin add wb");
      }
      if (event.getActivityType() == StreamedMedia.AUDIO_ACTIVITY_TYPE)
      {
        System.out.println("add audio");
        m_audioManager = new AudioManager();
        m_audioManager.connectAudio(m_session, m_place);
        mainVideoPanel.add(m_audioManager.layoutAudioUI(),BorderLayout.SOUTH);
        System.out.println("fin add audio");
      }
      if (event.getActivityType() == StreamedMedia.VIDEO_ACTIVITY_TYPE)
      {
        System.out.println("add video");
        m_videoManager = new VideoManager();
        m_videoManager.connectVideo(m_session, m_place);
        mainVideoPanel.add(m_videoManager.layoutVideoUI(),BorderLayout.CENTER);
        mainEastPanel.add(encapsulate(mainVideoPanel, border_color, false, false, true, false),BorderLayout.NORTH);
        m_videoManager.startVideoDevices();                        
        System.out.println("fin add video");
      }
    }
  }
	  
    class ValidButtonHandler implements ActionListener
    {
	public void actionPerformed(ActionEvent	event)
        {
            permModerator.removeAll();
            permModerator.dispose();
        }
    }
    
    class CancelButtonHandler implements ActionListener
    {
	public void actionPerformed(ActionEvent	event)
        {
            permModerator.removeAll();
            permModerator.dispose();
        }
    }
    
  /** 
   * A listener for participant list events.
   */
  class ParticipantListListener extends AwarenessViewAdapter
  {
    boolean firstTime = true;
    boolean moderator = false;
    
    public void usersAdded(AwarenessViewEvent event)
    {
        
        if (firstTime)
      {
        firstTime = false;
        numUsersInPlace = event.getUsers().length;
        setLabel(numUsersInPlace);
        if (numUsersInPlace == 1) becomeModerator(event);
        //else waitPermission(event);
      }
        else
        {
            numUsersInPlace++;
            setLabel(numUsersInPlace);
            if (moderator) askPermission(event);
        }			
    }

    public void becomeModerator(AwarenessViewEvent event)
    {
      /*  moderator=true;
        //moderatorName = new STExtendedAttribute(3,true);
        //myself_in_Place.changeAttribute(moderatorName);
        Frame infoModerator = new Frame();
        actualUser = event.getUsers()[0];
        System.out.println("nom utilisateur/moderateur "+actualUser.getDisplayName());
        Label infotextModerator = new Label("You are the moderator of this place.");
        infotextModerator.setFont(new Font("Dialog",Font.PLAIN,15));
        infotextModerator.setAlignment(infotextModerator.CENTER);
        infoModerator.add(infotextModerator);
        infoModerator.addWindowListener(new EcouteurQuitterFrame());    
        infoModerator.setSize(240,100);
        infoModerator.setLocation(1400,200);
        infoModerator.show();
      */
    }
    
    public void askPermission(AwarenessViewEvent event)
    {        
      /*  permModerator = new Frame();
        Panel permPanel = new Panel(new BorderLayout());
        permModerator.add(permPanel);
        Panel reppermPanel = new Panel(new FlowLayout());
        permPanel.add(reppermPanel,BorderLayout.SOUTH);
        users = event.getUsers();
        user = users[0];
        System.out.println("personne connectee "+users[0].getDisplayName());
        System.out.println("personne connectee "+users[users.length-1].getDisplayName());
        Label permtextModerator = new Label(user.getDisplayName()+" is here. Do you accept the connection?");
        permtextModerator.setFont(new Font("Dialog",Font.PLAIN,15));
        permtextModerator.setAlignment(permtextModerator.CENTER);
        Button validButton = new Button("Accept");
        Button cancelButton = new Button("Cancel");
        permPanel.add(permtextModerator,BorderLayout.CENTER);
        validButton.addActionListener(new ValidButtonHandler());
        cancelButton.addActionListener(new CancelButtonHandler());
        reppermPanel.add(validButton);
        reppermPanel.add(cancelButton);
        permModerator.addWindowListener(new EcouteurQuitterFrame());  
        permModerator.setSize(350,120);
        permModerator.setLocation(1400,300);
        permModerator.show();
      */
    }
    
    public void waitPermission(AwarenessViewEvent event)
    {
        Frame waitModerator = new Frame();
        Label waittextModerator = new Label("Wait autorisation to continue.");   
        waittextModerator.setFont(new Font("Dialog",Font.PLAIN,15));
        waittextModerator.setAlignment(waittextModerator.CENTER);
        waitModerator.add(waittextModerator);
        waitModerator.addWindowListener(new EcouteurQuitterFrame());  
        waitModerator.setSize(200,80);
        waitModerator.setLocation(1400,300);
        users = event.getUsers();
        user = users[0];
        System.out.println("nom user 0: " + user.getDisplayName());
        System.out.println("nom user fin: " + users[users.length-1].getDisplayName());
        System.out.println(" surnom de base user 0: " + user.getNickName());
        System.out.println(" surnom de base user fin: " + users[users.length-1].getNickName());
        waitModerator.show();
    }
    
    public void usersRemoved(AwarenessViewEvent event)
    {
      numUsersInPlace--;
      setLabel(numUsersInPlace);				
    }
      		
    private void setLabel(int numOfPeople)
    {
      m_PeopleNumLbl.setText(String.valueOf(numOfPeople));
      m_PeopleNumLbl.validate();
    }
  }
  
  class UserHandler extends UserInPlaceAdapter
  {
      public void attributeChanged(PlaceMemberEvent event)
      {
          switch (event.getAttributeKey())
          {
              case DATA_PERMISSION:
                  System.out.println(
                  "DATA PERMISSION CHANGED: " +
                  event.getAttribute().getBoolean());
                  break;
                  
               case AV_PERMISSION:
                  System.out.println(
                  "AUDIO/VIDEO PERMISSION CHANGED: " +
                  event.getAttribute().getBoolean());
                  break;
                  
               case MEETING_MODERATOR:
                  System.out.println(
                  "MODERATOR STATUS CHANGED: " +
                  event.getAttribute().getBoolean());
                  break;
          }
      }
      
      public void attributeRemoved(PlaceMemberEvent event)
      {
          if (MEETING_MODERATOR == event.getAttributeKey())
          {
              System.out.println("Moderator Status Changed: false");
          }
      }
  }
  
  /** 
   * Applet destroyed. Logout, stop devices, stop and 
   * unload the session.
   */
  public void destroy()
  {
    m_audioManager.stopAudioDevices();
    m_videoManager.stopVideoDevices();
    AppShare_Host.shutdown();
    m_appshareservice.stop();
    whiteboardService.stop();
    mainPanel.removeAll();       
    
    mainFrame.dispose();    

    try {
    JSObject win= (JSObject) JSObject.getWindow(this);
    win.eval( "self.close();");
    }
    //catch( JSException.InvocationTargetException ite) { }
    catch( Exception e) {System.out.println("error closing ie");}

    //m_place.leave();
//    m_place.close();
//    m_comm.logout();    
//    if (m_session != null) 
//    {
//      m_session.stop();
//      m_session.unloadSession();
//    }
  }  
}
