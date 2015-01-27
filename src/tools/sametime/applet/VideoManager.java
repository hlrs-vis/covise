import java.awt.*;
import java.awt.event.*;
import java.applet.*;

import com.lotus.sametime.core.comparch.*;
import com.lotus.sametime.places.Place;
import com.lotus.sametime.conference.*;
import com.lotus.sametime.streamedmedia.*;
import com.lotus.sametime.streamedmedia.event.*;

public class VideoManager  
{
  private static final String	PLAY = "Play";
  private static final String	PAUSE = "Pause";
  private static final String	RECORDER = "Home";
  private static final String	PLAYER = "Speaker";
													
  private StreamedMediaInteractive m_multimediaController;	
  private VideoController m_videoPlayer;
  private VideoController m_videoRecorder;

  private Panel m_videoPanel;
  private Panel	m_videoView;
  private VideoPanel m_videoPlayerControl;
  private VideoPanel m_videoRecorderControl;
  private VideoPanel saveVideoRecorder;

  protected static VideoPanel	m_currentView;
  protected Button m_playPauseButton;
  protected Choice m_recorderPlayerChoice;
  
  /** 
   * Connects the video manager to the session and place.
   */	
  public void connectVideo(STSession session, Place meetingPlace)
  {				
    StreamedMediaService streamed_media_service =
         (StreamedMediaService)session.getCompApi
                      (StreamedMediaService.COMP_NAME);
		
    m_multimediaController = 
      streamed_media_service.getStreamedMediaForPlace(meetingPlace);
    
    try
    {
      setupVideoDevices();
			
      m_videoRecorder = m_multimediaController.getVideoRecorder();
      m_videoPlayer = m_multimediaController.getVideoPlayer();
    }
    catch (Exception exception)
    {
      exception.printStackTrace();
    }			
  }
	
  /** 
   * Setup the video devices for work.
   */
  protected void setupVideoDevices()
  {
    try
    {									
      m_multimediaController.initVideo(null, null);
    }
    catch (StreamedMediaException	exception)
    {
      exception.printStackTrace();
    }	
  }
  
  /** 
   * Layout the video manager panel.
   */
  public Panel layoutVideoUI()
  {
    m_videoPanel = new Panel(new BorderLayout());
    m_videoView = new Panel(new CardLayout());
    m_videoPlayerControl = new VideoPanel(m_videoPlayer);

    m_videoRecorderControl = new VideoPanel(m_videoRecorder);
		
    Panel puseVideoPanel = new Panel();
    puseVideoPanel.setBackground(new Color(0xdedede) ); // Color.yellow);
    puseVideoPanel.add(new Label("Pause"));
		
    //m_videoPlayerControl.setActive(true);
    //m_videoRecorderControl.setActive(true);
		
    m_videoView.add(m_videoRecorderControl, RECORDER);
    m_videoView.add(m_videoPlayerControl,PLAYER);
    m_videoView.add(puseVideoPanel,PAUSE);
    m_videoView.setBackground( new Color( 0xdedede) );
	
    m_currentView = m_videoRecorderControl;
    ((CardLayout)m_videoView.getLayout()).show(m_videoView,RECORDER);
    m_videoPanel.add(m_videoView, BorderLayout.CENTER);
		
    VideoMode vidoe_mode = new VideoMode();
    m_playPauseButton = new Button(PAUSE);
    m_playPauseButton.setBackground(new Color(0xdedede) );
    m_playPauseButton.addActionListener(vidoe_mode);
    m_recorderPlayerChoice = new Choice();
    m_recorderPlayerChoice.setBackground(new Color(0xdedede) );
    m_recorderPlayerChoice.add(RECORDER);
    m_recorderPlayerChoice.add(PLAYER);
    m_recorderPlayerChoice.addItemListener(vidoe_mode);
		
    Panel pnl = new Panel(new FlowLayout(FlowLayout.CENTER));
    pnl.add(m_playPauseButton);
    pnl.add(m_recorderPlayerChoice);
    m_videoPanel.add(pnl, BorderLayout.SOUTH);
    m_videoPanel.setBackground( new Color( 0xdedede) );
    m_videoPanel.validate();
		
    startVideoDevices();
		
    return m_videoPanel;
  }
	
  /** 
   * Start the video devices.
   */
  public void startVideoDevices()
  {
    m_videoPlayerControl.setPlaying(true);
    m_videoRecorderControl.setPlaying(true);
    m_videoPlayerControl.setActive(true);
    m_videoRecorderControl.setActive(true);
  }
	
  /** 
   * Stop the video devices.
   */
  public void stopVideoDevices()
  {
    m_videoPlayerControl.setActive(false);
    m_videoRecorderControl.setPlaying(false);
    m_videoPlayerControl.setActive(false);
    m_videoRecorderControl.setActive(false);
  }

  /** 
   * Label displayed when sharing launched.
   */
  public Label shareLabel()
  {
      Label messLabel = new Label("VIDEO OFF");
      return messLabel;
  }
  
  /** 
   * Stop the local video.
   */
  public void stopLocalVideoDevices()
  {
    m_videoRecorderControl.setMonitorEnabled(false);
    m_videoRecorderControl.setStreamEnabled(false);
    //m_videoRecorderControl.VideoPanelPaused(shareLabel());
  }
  
  /** 
   * This class toggle between the player and recorder video.
   */
  class VideoMode implements ActionListener,ItemListener 
  {
    public void actionPerformed(ActionEvent event)
    {	
      if (m_currentView.isActive())
      {
        m_currentView.setActive(false);
        m_playPauseButton.setLabel(PLAY);
      }
      else 
      {
        m_currentView.setActive(true);
	m_playPauseButton.setLabel(PAUSE);
      }
    }

    public void itemStateChanged(ItemEvent event)
    {
      if (((String)event.getItem()).equals(PLAYER))
      {
        m_currentView = m_videoPlayerControl;
        ((CardLayout)m_videoView.getLayout()).show(m_videoView,PLAYER);
        m_currentView.setActive(true);
      }
      else
      {
        m_currentView = m_videoRecorderControl;
        ((CardLayout)m_videoView.getLayout()).show(m_videoView, RECORDER);
        m_currentView.setActive(true);
      }	
    }
  }
}
