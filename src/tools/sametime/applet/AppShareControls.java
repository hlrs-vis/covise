import	com.lotus.sametime.appshare.AppShare;
import	com.lotus.sametime.appshare.ShareWindow;

import	com.lotus.sametime.appshare.event.AppShareAdapter;
import	com.lotus.sametime.appshare.event.AppShareEvent;

import	com.lotus.sametime.appshare.event.ShareWindowAdapter;
import	com.lotus.sametime.appshare.event.ShareWindowEvent;

import  java.awt.*;

import	java.awt.event.*;
import	java.awt.event.ActionEvent;
import	java.awt.event.ActionListener;
import	java.awt.event.ItemEvent;
import	java.awt.event.ItemListener;

/**
 * This class provides user interfaces controls for manipulating
 * the AppShare object.  Including are controls for sharing a 
 * Window, Frame or Desktop as well as controls to Drive a remotely
 * shared application and to allow others to drive the locally
 * shareed application.  The enabled/disabled state of the UI controls
 * are modified based on the current sharing status as well as the
 * permissions granted to the local user.
 */
public class AppShareControls extends Panel
{
	/**
	 * GUI controls.
	 */
	public static	MotifButton		Share_Window_Button		= null;
	private Button		Share_Desktop_Button	= null;
	private	Button		Share_Frame_Button		= null;
	private	MotifButton		Stop_Sharing_Button		= null;
	
	private PopupMenu	Share_Window_Popup		= null;
	
	private	MotifButton	Drive_Checkbox			= null;
	public static	MotifButton	Allow_Drive_Checkbox	= null;

	/**
	 * Boolean value representing whether or not this user
	 * has permission to drive (control) the remotely shared
	 * application.
	 */
	private	boolean		Local_Permission_To_Drive	= false;
	
	/**
	 * Boolean value representing whether or not there is a
	 * appshare host actively sharing.
	 */
	private	boolean		Host_Is_Present	= false;
	
	/**
	 * Boolean value representing whether or not this user has
	 * data permissions in the meeting.
	 */
	private	boolean		Permission_Enabled	= true;
	
	/**
	 * The currently shared window.
	 */
	public static	ShareWindow		Share_Window	= null;
	
	/**
	 * The ShareWindowHandler responsible for responding to
	 * ShareWindowEvents for the current ShareWindow.
	 */
	private	ShareWindowHandler	Share_Window_Handler	= null;

	/**
	 * The AppShare object.
	 */
	private	AppShare	App_Share	= null;
	
	/**
	 * An object used for synchronization.
	 */
	private	Object	Synchronizer	= null;

	/**
	 * The AppShareHandler responsible for responding to
	 * AppShareEvents received from the AppShare object.
	 */
	private AppShareHandler	AppShare_Handler	= null;
	
	/**
	 * The STSamplePermissionsManager for this meeting.
	 */
//	private	STSamplePermissionsManager	Permissions_Manager	= null;
	
	/**
	 * Constructs an STSampleAppShareControls object for the 
	 * given AppShare.
	 * 
	 * @param app_share
	 *  The AppShare object to be controlled.
	 * 
	 * @param permissions_manager
	 *  The STSamplePermissionsManager used to determine what
	 *  permissions this user has so that GUI controls can
	 *  be enabled/disabled appropriately.
	 */
	public AppShareControls(AppShare app_share)
	{
                App_Share = app_share;
		//Permissions_Manager = permissions_manager;

		Synchronizer = new Object();

		//
		// Create/layout the UI components.
		//
		layoutUI();

		//
		// Add an AppShareListener to the AppShare object. 
		//
		App_Share.addAppShareListener(getAppShareHandler());
	}
	
	/**
	 * Shuts down this STSampleAppShareControls.
	 */
	public void shutdown()
	{	
		if (null != Share_Window)
		{
			Share_Window.unShare();
			Share_Window.removeShareWindowListener(Share_Window_Handler);
			Share_Window = null;
		}
		
		if (null != App_Share)
		{
			App_Share.removeAppShareListener(getAppShareHandler());
		}
	}
		
	/** 
	 * Constructs the GUI components for this
	 * STSampleAppShareControls object and constructs
	 * and initializes the layout.
	 */
	protected void layoutUI()
	{
		//
		//	Create the share window listener.
		//
		Share_Window_Handler = new ShareWindowHandler();
       
                //
		//	Create the share window button and menu item.
		//
		Share_Window_Button = new MotifButton( MeetingLauncher.stopAppSharingIcon, 
		                                               MeetingLauncher.startAppSharingIcon, 				                                                               "   Stop Sharing", "Share Application", 170);
          
                ShareWindowButtonHandler swb = new ShareWindowButtonHandler();
                swb.setButton( Share_Window_Button );
                Share_Window_Button.addMouseListener(swb);
                Share_Window_Button.addMouseMotionListener(swb);		               		


                Share_Frame_Button = new Button("Share Frame");
		Share_Desktop_Button = new Button("Share Desktop");
		
		Share_Window_Popup = new PopupMenu();
		Share_Window_Button.add(Share_Window_Popup);	        


		Drive_Checkbox = new MotifButton( MeetingLauncher.box_checked, 
		                                         MeetingLauncher.box, 
		                                         "Take Control", "Take Control", 140);
							 
                DriveButtonHandler dw = new DriveButtonHandler();
                dw.setButton( Drive_Checkbox );
                Drive_Checkbox.removeBorders();
                Drive_Checkbox.addMouseListener(dw);
                Drive_Checkbox.addMouseMotionListener(dw);
                
		
                Allow_Drive_Checkbox = new MotifButton( MeetingLauncher.box_checked, 
		                                                 MeetingLauncher.box, 
								 "Allow Control", "Allow Control", 150);
								 
                AllowDriveButtonHandler aw = new AllowDriveButtonHandler();
                aw.setButton( Allow_Drive_Checkbox );
                Allow_Drive_Checkbox.removeBorders();
                Allow_Drive_Checkbox.addMouseListener(aw);
                Allow_Drive_Checkbox.addMouseMotionListener(aw);
               		
                		
		Share_Frame_Button.addActionListener(new ShareFrameButtonHandler());
		Share_Desktop_Button.addActionListener(new ShareDesktopButtonHandler());		
                
		setLayout(new FlowLayout());
		
                add(Share_Window_Button);                
                                                         		
	        add(Drive_Checkbox);
		add(Allow_Drive_Checkbox);                                
                
	}
	
	/**
	 * Returns this STSampleAppShareControls viewable Java 
	 * AWT Component.
	 * 
	 * @return Component
	 *  The viewable Component.
	 */
	public Component getViewableComponent()
	{
		return (this);
	}
	
	/** 
	 * Enables or disables the component.
	 *
	 * @param is_enabled
	 *  "true" if the component is enabled.
	 *  "false" otherwise.
	 */		
	protected void setPermissionEnabled(boolean is_enabled)
	{
		synchronized (Synchronizer)
		{
			Permission_Enabled = is_enabled;

			if (is_enabled == false)
			{
				//
				//	If anything is currently being shared, stop sharing.
				//
				stopSharing();

				//
				//	Turn off driving.
				//
				App_Share.controlRemoteApp(false);

				//
				//	Disable and unpress all UI components that weren't taken
				//	care of in stopSharing().
				//
				Share_Window_Button.setEnabled(false);
				Share_Frame_Button.setEnabled(false);
				Share_Desktop_Button.setEnabled(false);
				Drive_Checkbox.setEnabled(false);
				Allow_Drive_Checkbox.setEnabled(false);
				Allow_Drive_Checkbox.setState(false);
				//Stop_Sharing_Button.setEnabledfalse);
                                MeetingLauncher.mainPanel.validate();

			}
			else
			{
				//
				//	Depending on whether anything is being shared, enable the
				//	appropriate UI components.
				//
				if ((false == Host_Is_Present) &&
					(true == App_Share.isSharingAvailable()))
				{
					//
					// AppShare is available and there isn't a host.
					// enable the sharing controls.
					//
					Share_Window_Button.setEnabled(true);
					MeetingLauncher.mainPanel.validate();
					Share_Frame_Button.setEnabled(true);
					Share_Desktop_Button.setEnabled(true);
				}
				else
				{
					//
					// Enable/disable the Drive Checkbox based on
					// our permissions.
					//
					if (true == App_Share.isRemoteControlAvailable())
					{
						Drive_Checkbox.setEnabled(Local_Permission_To_Drive);
					}
					else
					{
						Drive_Checkbox.setEnabled(false);
						Drive_Checkbox.setState(false);
					}
				}
			}
		}
	}

        
        
        protected boolean member(String s1, String s2)
        {
            String string = new String(s1);
            for (int i=0;i<string.length();i++)
            {
                if (string.startsWith(s2,i)) return true;
            }
            return false;
        }
        
        protected void moveRightMainFrame()
        {
            //MeetingLauncher.m_videoManager.stopLocalVideoDevices();
            //MeetingLauncher.mainFrame.setLocation(1150,-25);
            //swapMainPanel();
            //MeetingLauncher.mainPanel.validate();
        }
        
        protected void moveLeftMainFrame()
        {
            //swapMainPanel();
            //MeetingLauncher.mainPanel.validate();
            //MeetingLauncher.m_videoManager.startVideoDevices();
            //MeetingLauncher.mainFrame.hide();
            //MeetingLauncher.mainFrame.show();
            //MeetingLauncher.mainFrame.setLocation(1,-25);
        }
        
        protected void swapMainPanel()
        {
            //Panel rightPanel;
            //Panel leftPanel;
            //rightPanel = (Panel)MeetingLauncher.mainPanel.getComponent(0);
            //leftPanel = (Panel)MeetingLauncher.mainPanel.getComponent(1);
            //MeetingLauncher.mainPanel.removeAll();
            //MeetingLauncher.mainPanel.add(leftPanel);
            //MeetingLauncher.mainPanel.add(rightPanel);
        }
        
	/**
	 * This method populates a menu with a list of windows.
	 *
	 * @param menu
	 *  The menu to be populated with the list of windows.
	 *
	 * @return ShareWindow[]
	 *  The array of share windows used to populate the menu.
	 */
	protected void populateShareWindowMenu(Menu	menu)
	{
		ShareWindow[]	share_windows = null;

		//
		//	Retrieve the list of windows from the core component.
		//
		share_windows = App_Share.createWindowArray();

		if ((null != share_windows) && (share_windows.length > 0))
		{
			ShareWindow			window = null;
			CheckboxMenuItem	menu_item = null;
			boolean				item_checked = false;

			//
			//	Clear out the previous contents of the menu.
			//
			menu.removeAll();

                        //
                        // 1.run: add Microsoft Office products to the menu 
			// 2.run: add other windows
			//
                	for( int sort=0; sort<2; sort++ ) {

			    //
			    //	Add the window titles to the menu.
			    //
			    for (int count = 0; count < share_windows.length; count++)
			    {
                                    window = share_windows[count];
                                    System.out.println("nom appli" + window.getName());

                                    if (    ( sort==1 &&
                                            (  member(window.getName(),"Microsoft Word") || 
					       member(window.getName(),"Microsoft PowerPoint") ||
					       member(window.getName(),"Microsoft Excel")))       
					||
                                            ( sort==0 && 
                                             ! (  member(window.getName(),"COVISE Conference") ||
						  member(window.getName(),"Microsoft Word") || 
						  member(window.getName(),"Microsoft PowerPoint") ||
						  member(window.getName(),"Microsoft Excel"))) )
                                	{
					menu_item = new CheckboxMenuItem(window.getName());
					menu_item.addItemListener(new ShareWindowMenuItemHandler(window));
                                	if( sort==1 &&
                                             ( member(window.getName(),"Microsoft Word") || 
					       member(window.getName(),"Microsoft PowerPoint") ||
					       member(window.getName(),"Microsoft Excel")) )     {

				        	 menu_item.setFont( new Font("Dialog", Font.BOLD, 12)); 
					}
					menu.add(menu_item);

					//
					//	If this is the currently shared window, checkmark it.
					//
					if (Share_Window != null)
				            {
				            if ((false == item_checked) && (window.getHandle() == Share_Window.getHandle()))
						{
						menu_item.setState(true);
						item_checked = true;
						}
				            }
                                	}
			    }
                            
			    //
			    // add Seperator
			    //
                            if( sort==0 ) {
                               menu_item = new CheckboxMenuItem("-");                          
                               menu.add(menu_item);
                            }
                	}
		}
		else
		{
			System.out.println("AppShareControls.populateShareWindowMenu() " + " No windows returned");
		}
	}

        /** 
	* This method returns false if an application is currently shared, otherwise true
	* It is called from the class MeetingLauncher and therefore static
	*/
        public static boolean noApplicationShared() {
          return( Share_Window_Button.isVisible()  || 
                  Allow_Drive_Checkbox.isVisible() );
	}

        /** 
	* This method updates the window size of the UI
	* It calls methods from the class MeetingLauncher 
	*/
        public void update_UI_Size() {
          if( !MeetingLauncher.isWhiteboardActive() ) {
             if( noApplicationShared() ) {
               MeetingLauncher.minimize_UI();
             }
             else {
               MeetingLauncher.maximize_UI();
             }
          }
        }
             
        /**
	 * This method stops sharing the currently shared window and 
	 * updates the UI components accordingly.
	 */
	protected void stopSharing()
	{
		//
		//	If something is already being shared, stop it.
		//
		if (null != Share_Window)
		{
			Share_Window.unShare();
			Share_Window.removeShareWindowListener(Share_Window_Handler);
			Share_Window = null;
		}

		//
		//	Check/uncheck the appropriate buttons and menu items.
		//
		Share_Window_Button.setEnabled(true);
		Share_Frame_Button.setEnabled(true);
		Share_Desktop_Button.setEnabled(true);
		Drive_Checkbox.setEnabled(false);
		Drive_Checkbox.setState(false);
		Allow_Drive_Checkbox.setEnabled(false);
		Allow_Drive_Checkbox.setState(false);
                MeetingLauncher.mainPanel.validate();
		//Stop_Sharing_Button.setEnabledfalse);
	}
	
	/**
	 * Returns the AppShareHandler object.
	 * 
	 * @return AppShareHandler
	 *  The AppShareHandler class for this STSampleAppShareControls.
	 */
	private AppShareHandler getAppShareHandler()
	{
		//
		// If the AppShareHandler hasn't been created do
		// so now.
		//
		if (null == AppShare_Handler)
		{
			AppShare_Handler = new AppShareHandler();
		}
		
		return (AppShare_Handler);
	}

	/**
	 * Inner class to handle ActionEvents from the Share Window
	 * button.
	 */
	class ShareWindowButtonHandler extends MouseAdapter implements MouseMotionListener
	{
		/**
		 * Called when the Share Window button has been 
		 * pressed.  We must populate the share window menu
		 * and show it.
		 * 
		 * @param event
		 *  The ActionEvent received from the Share Window button.
		 */
		MotifButton _allowButton; 

        	public void setButton( MotifButton allowButton) {          
        	  _allowButton = allowButton;          
        	}         	                        

		public void mouseDragged(MouseEvent e) { }

		public void mouseMoved(MouseEvent e) {}

        	public void mouseReleased(MouseEvent e) {}

        	public void mousePressed(MouseEvent e) {
		   synchronized (Synchronizer) {

                	    if ( _allowButton.getState() ) {
                        	_allowButton.setState(false);
                	        stopSharing();              

                	     }
                	   else {                	                       				

				//
				//	Populate the popup menu with the window list.
				//
				populateShareWindowMenu(Share_Window_Popup);

				//
				//	Display pop up on the share window button.
				//
				Share_Window_Popup.show(Share_Window_Button,
										0,
										37);

                	   }
                	   _allowButton.repaint();

                   }	
		}
	}
	
	/**
	 * Inner class used to handle ItemEvents from the 
	 * Share Window menu.
	 */
	class ShareWindowMenuItemHandler implements ItemListener
	{
		private	ShareWindow	Menu_Share_Window = null;

		/**
		 * Constructs a ShareWindowMenuItemHandler for the given 
		 * ShareWindow.
		 * 
		 * @param menu_share_window
		 *  The ShareWindow associated with this menu item handler.
		 */
		public ShareWindowMenuItemHandler(ShareWindow menu_share_window)
		{
			Menu_Share_Window = menu_share_window;
		}

		/**
		 * Called when a menu item has been checked/unchecked 
		 * from the share window menu.
		 * 
		 * @param event
		 *  The ItemEvent received from the menu item.
		 */
		public void itemStateChanged(ItemEvent event)
		{
			MenuItem item = (MenuItem)event.getSource();
			
			synchronized (Synchronizer)
			{
				//
				//	If something is already being shared, stop it.
				//
				if (null != Share_Window)
				{
					Share_Window.unShare();
					Share_Window.removeShareWindowListener(Share_Window_Handler);
					Share_Window = null;
				}

				if (event.getStateChange() == ItemEvent.SELECTED)
				{
					//
					//	Check/uncheck the appropriate buttons and menu items.
					//
					Share_Frame_Button.setEnabled(false);
					Share_Desktop_Button.setEnabled(false);
					//Share_Window_Button.setEnabledfalse);

					//
					//	Share the window.
					//
					if (null != Menu_Share_Window)
					{
						Share_Window = Menu_Share_Window;

						Share_Window.addShareWindowListener(Share_Window_Handler);

						//if (Share_Window.share(
						//			ShareWindow.HIGH_PRIORITY,
						//			ShareWindow.EMBEDDED_CONTROL_BUTTONS,
						//			null,
						//			"Stop Sharing",
						//			"Allow Control"))
                                                
                                                if (Share_Window.share(
									ShareWindow.HIGH_PRIORITY,
									ShareWindow.NO_CONTROL_BUTTONS,
                                                                        null,
                                                                        null,
                                                                        null))
                                                
                                                
						{
							//
							//	Enable/disable the appropriate buttons and
							//	menu items.
							//
                                                        moveRightMainFrame();
							//Stop_Sharing_Button.setEnabledtrue);
							Allow_Drive_Checkbox.setEnabled(true);
							Drive_Checkbox.setEnabled(false);
                                                        Share_Window_Button.setState(true);
                                                        MeetingLauncher.mainPanel.validate();
						}
						else
						{
							System.out.println("Error attempting to share a window");
							Share_Window.removeShareWindowListener(Share_Window_Handler);
							Share_Window = null;
						}
					}
				}

				//
				//	If nothing is being shared, reset the rest of the UI.
				//
				if (null == Share_Window)
				{
					Share_Frame_Button.setEnabled(true);
					Share_Desktop_Button.setEnabled(true);
					Share_Window_Button.setEnabled(true);
					
					Drive_Checkbox.setEnabled(false);
					//Stop_Sharing_Button.setEnabledfalse);
					Allow_Drive_Checkbox.setEnabled(false);
					MeetingLauncher.mainPanel.validate();
				}
			}
		}
	}

	/**
	 * Inner class used to handle ActionEvents from the Share
	 * Frame button.
	 */
	class ShareFrameButtonHandler implements ActionListener
	{		
		/**
		 * Called when the Share Frame button has been 
		 * pressed.  
		 * 
		 * @param event
		 *  The ActionEvent received from the Share Frame button.
		 */
		public void actionPerformed(ActionEvent	event)
		{
			synchronized (Synchronizer)
			{
				//
				//	If something is already being shared, stop it.
				//
				if (null != Share_Window)
				{
					Share_Window.unShare();
					Share_Window.removeShareWindowListener(Share_Window_Handler);
					Share_Window = null;
				}

				Share_Window = App_Share.createSharedFrame("Sharing with a Frame");

				if (null != Share_Window)
				{
					Share_Window.addShareWindowListener(Share_Window_Handler);

					if (Share_Window.share(
											ShareWindow.HIGH_PRIORITY,												   
											ShareWindow.EMBEDDED_CONTROL_BUTTONS,
											null,
											"Stop Sharing",
											"Allow Control"))
					{				
						//
						//	Check/uncheck the appropriate buttons and menu
						//	items.
						//
						Share_Frame_Button.setEnabled(false);
						//Share_Window_Button.setEnabledfalse);
						Share_Desktop_Button.setEnabled(false);		

						//
						//	Enable/disable the appropriate buttons and
						//	menu items.
						//
						Allow_Drive_Checkbox.setEnabled(true);
						//Stop_Sharing_Button.setEnabledtrue);
						Drive_Checkbox.setEnabled(false);
						Drive_Checkbox.setState(false);
 						MeetingLauncher.mainPanel.validate();
							
						// FIX Share_Window.remoteControl(
						//	Allow_Others_To_Drive_State.getState());
															  
					}
					else
					{
						System.out.println("Error attempting to share a frame");

						//
						//	Check/uncheck the appropriate buttons and menu
						//	items.
						//
						Share_Frame_Button.setEnabled(true);
						Share_Window.removeShareWindowListener(Share_Window_Handler);
						Share_Window = null;
					}
				}
			}
			
			//
			//	If nothing is being shared, reset the rest of the UI.
			//
			if (null == Share_Window)
			{
				Share_Frame_Button.setEnabled(true);
				Share_Desktop_Button.setEnabled(true);
				Share_Window_Button.setEnabled(true);
					
				Drive_Checkbox.setEnabled(false);
				//Stop_Sharing_Button.setEnabledfalse);
				Allow_Drive_Checkbox.setEnabled(false);
				MeetingLauncher.mainPanel.validate();
			}
		}
	}

	/**
	 * Inner class used to handle ActionEvents from the Share Desktop
	 * button.
	 */
	class ShareDesktopButtonHandler implements ActionListener
	{	
		/**
		 * Called when the Share Desktop button has been 
		 * pressed.  
		 * 
		 * @param event
		 *  The ActionEvent received from the Share Desktop button.
		 */
		public void actionPerformed(ActionEvent	event)
		{
			synchronized (Synchronizer)
			{
				//
				//	If something is already being shared, stop it.
				//
				if (null != Share_Window)
				{
					Share_Window.unShare();
					Share_Window.removeShareWindowListener(Share_Window_Handler);
					Share_Window = null;
				}

				Share_Window = App_Share.createDesktopWindow();

				if (null != Share_Window)
				{
					Share_Window.addShareWindowListener(Share_Window_Handler);

					if (Share_Window.share(
								ShareWindow.HIGH_PRIORITY,
								ShareWindow.FRAMED_CONTROL_BUTTONS, //EMBEDDED_CONTROL_BUTTONS,
								"Sharing Desktop",
								"Stop Sharing",
								"Allow Control"))
					{
						//
						//	Check/uncheck the appropriate buttons and menu
						//	items.
						//
						Share_Frame_Button.setEnabled(false);
						//Share_Window_Button.setEnabledfalse);
						Share_Desktop_Button.setEnabled(false);		

						//
						//	Enable/disable the appropriate buttons and
						//	menu items.
						//
						Allow_Drive_Checkbox.setEnabled(true);
						//Stop_Sharing_Button.setEnabledtrue);
						Drive_Checkbox.setEnabled(false);
						Drive_Checkbox.setState(false);
						MeetingLauncher.mainPanel.validate();
					}
					else
					{
						System.out.println("Error attempting to share desktop");

						//
						//	Check/uncheck the appropriate buttons and menu
						//	items.
						//
						//
						//	Check/uncheck the appropriate buttons and menu
						//	items.
						//
						Share_Frame_Button.setEnabled(true);
						Share_Window_Button.setEnabled(true);
						Share_Desktop_Button.setEnabled(true);		

						//
						//	Enable/disable the appropriate buttons and
						//	menu items.
						//
						Allow_Drive_Checkbox.setEnabled(false);
						Allow_Drive_Checkbox.setState(false);
						//Stop_Sharing_Button.setEnabledfalse);
						Drive_Checkbox.setEnabled(false);
						Drive_Checkbox.setState(false);
						
						MeetingLauncher.mainPanel.validate();	
						Share_Window.removeShareWindowListener(Share_Window_Handler);
						Share_Window = null;
					}
				}
				else
				{
					System.out.println("Error attempting to create a desktop window");

					Share_Frame_Button.setEnabled(true);
					Share_Desktop_Button.setEnabled(true);
					Share_Window_Button.setEnabled(true);
					
					Drive_Checkbox.setEnabled(false);
					//Stop_Sharing_Button.setEnabledfalse);
					Allow_Drive_Checkbox.setEnabled(false);
					MeetingLauncher.mainPanel.validate();
				}
			}
		}
	}
	

	/**
	 * Inner class used to handle ActionEvents from the 
	 * Stop Sharing button.
	 */
    
     class StopSharingButtonHandler extends MouseAdapter implements MouseMotionListener{
                 /**
		 * Called when the Stop Sharing button has been 
		 * pressed.  
		 * 
		 * @param e
		 *  The Mouse event received from the Stop Sharing button.
		 */ 

        	public void mousePressed(MouseEvent e) {
        	   synchronized (Synchronizer) {			
		     stopSharing(); 
        	   }
		}

		public void mouseDragged(MouseEvent e) { }

		public void mouseMoved(MouseEvent e) {}

        	public void mouseReleased(MouseEvent e) {}
     }
   
   
         /**
	 * Inner class used to handle ItemEvents from the 
	 * Drive checkbox.
	 */
	class DriveButtonHandler extends MouseAdapter implements MouseMotionListener
	{
		/**
		 * Called when the Allow Drive checkbox changes state.
		 * 
		 * @param e
		 *  The MouseEvent from the Allow Drive Checkbox.
		 */

		 MotifButton _driveButton; 

        	 public void setButton( MotifButton driveButton) {
        	  _driveButton = driveButton;
        	 } 

        	 public void mousePressed(MouseEvent e) {
        	   synchronized (Synchronizer) {

        		if ( _driveButton.getState() )
        		{
                	     _driveButton.setState(false);
                	     App_Share.controlRemoteApp(false);
        		}
        		else
        		{
                	      _driveButton.setState(true);
                	      App_Share.controlRemoteApp(true);       
        		}           
        		_driveButton.repaint();
        	  }
		 }

		 public void mouseDragged(MouseEvent e) {}

		 public void mouseMoved(MouseEvent e) {}

        	 public void mouseReleased(MouseEvent e) {}
	}

	/**
	 * Inner class used to handle ItemEvents from the Allow
	 * Drive checkbox.
	 */
	class AllowDriveButtonHandler extends MouseAdapter implements MouseMotionListener
	{
		/**
		 * Called when the Allow Drive checkbox changes state.
		 * 
		 * @param e
		 *  The MouseEvent from the Allow Drive Checkbox.
		 */

		 MotifButton _allowButton; 

        	 public void setButton( MotifButton allowButton) {
        	  _allowButton = allowButton;
        	 } 

        	 public void mousePressed(MouseEvent e) {         

        	    if ( _allowButton.getState() )
        	    {
                	 _allowButton.setState(false);
                	 Share_Window.allowRemoteControl(false);
        	    }
        	    else
        	    {
                	  _allowButton.setState(true);
                	  Share_Window.allowRemoteControl(true);       
        	    }           
        	    _allowButton.repaint();         
		 }

		 public void mouseDragged(MouseEvent e) {}

		 public void mouseMoved(MouseEvent e) {}

        	 public void mouseReleased(MouseEvent e) {}
	}

	/**
	 * Inner class used to handle AppShareEvents.
	 */
	class AppShareHandler extends AppShareAdapter
	{
		/**
		 * Called when the AppShare object is fully functional
		 * and available.
		 * 
		 * @param event
		 *  The AppShareEvent.
		 */
		public void appshareAvailable(AppShareEvent	event)
		{
//			Permissions_Manager.addPermissionManagerListener(new PermissionManagerHandler());
		}
		
		/**
		 * Called when the host has changed.
		 * 
		 * @param event
		 *  The AppShareEvent.
		 */
		public void currentHostChanged(AppShareEvent event)
		{
                        System.out.println("currenthostchanged");
			synchronized (Synchronizer)
			{
				Host_Is_Present = event.isHostPresent();

				//
				//	If someone is hosting, disable the share buttons.
				//	Otherwise, enable the share buttons, as long as
				//	permissions are enabled.
				//
				if (true == Host_Is_Present)
				{
					//
					//	Enable/disable the appropriate buttons and menu items.
					//
					Share_Frame_Button.setEnabled(false);
					Share_Desktop_Button.setEnabled(false);
					
					
					//Stop_Sharing_Button.setEnabledfalse);
					Allow_Drive_Checkbox.setEnabled(false);
					Allow_Drive_Checkbox.setState(false);
					
					if (true == App_Share.isRemoteControlAvailable())
					{
						Drive_Checkbox.setEnabled(Local_Permission_To_Drive && Permission_Enabled);
                                                Share_Window_Button.setEnabled(Local_Permission_To_Drive && Permission_Enabled);
					}
					else
					{
                                                Share_Window_Button.setEnabled(false);
						Drive_Checkbox.setEnabled(false);
						Drive_Checkbox.setState(false);						
					}
                                                                    
				}
				else
				{
					//
					//	Turn off the local user's permission to drive.  When a
					//	new user begins hosting, he will select whether other
					//	users can drive.
					//
					Local_Permission_To_Drive = false;

					//
					//	Check/uncheck the appropriate buttons and menu items.
					//
					Drive_Checkbox.setEnabled(false);
					Drive_Checkbox.setState(false);
					//Stop_Sharing_Button.setEnabledfalse);
					Allow_Drive_Checkbox.setEnabled(false);
					Allow_Drive_Checkbox.setState(false);

					//
					//	Enable/disable the appropriate buttons and menu items.
					//
					if ((true == Permission_Enabled) &&
						(true == App_Share.isSharingAvailable()))
					{
						Share_Frame_Button.setEnabled(true);
						Share_Desktop_Button.setEnabled(true);
						Share_Window_Button.setEnabled(true);
						MeetingLauncher.mainPanel.validate();
					}
                                        
				}
				// check for update
                                update_UI_Size();
			}
		}
		
		/**
		 * Called when the permissions for driving the remotely
		 * shared application have changed.
		 * 
		 * @param event
		 *  The AppShareEvent.
		 */
		public void controlPermissionChanged(AppShareEvent event)
		{
                        Drive_Checkbox.setEnabled(true);
                        MeetingLauncher.mainPanel.validate();
			synchronized (Synchronizer)
			{
				Local_Permission_To_Drive = event.allowedToRemoteControl();
				
				if (true == Host_Is_Present)
				{
					if ((true == Local_Permission_To_Drive) &&
						(true == Permission_Enabled))
					{
						Drive_Checkbox.setEnabled(true);
						MeetingLauncher.mainPanel.validate();
					}
					else
					{
						//
						//	Stop controlling the remote app.
						//
						App_Share.controlRemoteApp(false);

						//
						//	Unpress and disable the drive button/menu item.
						//
						Drive_Checkbox.setEnabled(false);
						Drive_Checkbox.setState(false);
					}
				}
			}
		}
	}

	/**
	 * Inner class used for handling ShareWindowEvents from the
	 * current ShareWindow.
	 */
	class ShareWindowHandler extends ShareWindowAdapter
	{
		/**
		 * Called when the current share window has been 
		 * closed.
		 * 
		 * @param event
		 *  The ShareWindowEvent.
		 */
		public void windowClosed(ShareWindowEvent event)
		{
                        System.out.println("windowclosed");
                        moveLeftMainFrame();
			synchronized (Synchronizer)
			{
				if (null != Share_Window)
				{
					Share_Window.removeShareWindowListener(Share_Window_Handler);
					Share_Window = null;
					
					Share_Frame_Button.setEnabled(true);
					Share_Desktop_Button.setEnabled(true);
					Share_Window_Button.setEnabled(true);
					
					//Stop_Sharing_Button.setEnabledfalse);
					Allow_Drive_Checkbox.setEnabled(false);
					Allow_Drive_Checkbox.setState(false);
					Drive_Checkbox.setEnabled(false);
					Drive_Checkbox.setState(false);
					MeetingLauncher.mainPanel.validate();
				}
			}
		}

		/**
		 * Called when the local video mode is invalid/not supported
		 * by the AppShare host.
		 * 
		 * @param event
		 *  The ShareWindowEvent.
		 */
		public void invalidVideoMode(ShareWindowEvent event)
		{
			synchronized (Synchronizer)
			{
				stopSharing();
			}
		}
	}
	
	/**
	 * Inner class used for handling STSamplePermissionManagerEvents
	 * from the STSamplePermissionsManager.
	 */
//	class PermissionManagerHandler extends STSamplePermissionManagerAdapter
//	{
//		/**
//		 * Called when one of the permissions has changed for 
//		 * this user.
//		 * 
//		 * @param event
//		 *  The STSamplePermissionManagerEvent.
//		 */
//		public void permissionChanged(STSamplePermissionManagerEvent event)
//		{
//			if (STSamplePermissionsManager.DATA_PERMISSION == event.getPermissionType())
//			{
//				setPermissionEnabled(event.isPermissionEnabled());	
//			}
//		}
//	}
}
