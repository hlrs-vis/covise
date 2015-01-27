import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
			
import com.lotus.sametime.streamedmedia.VideoController;
import com.lotus.sametime.streamedmedia.event.*;

/** 
 * Video Panel providing UI access to a single video controller.
 */
public class VideoPanel extends Panel
{
  private VideoController m_videoController;
	
  private boolean m_active = false;
  private boolean m_playing = false;
		
  /** 
   * Video panel constructor.
   */
  public VideoPanel(VideoController videoController)
  {
    m_videoController = videoController;
    setLayout(new BorderLayout(10,5));
		
    add("Center", m_videoController.getViewableComponent());
  }
  
  public VideoPanel()
  {  
      new Panel();
      setLayout(new BorderLayout());
  }
	
  /** 
   * Set the video controller as paused.
   */
  public void VideoPanelPaused(Label videoScreenPaused)
  {
      removeAll();
      add("Center", videoScreenPaused);
  }  
  
  /** 
   * Set the video controller as active.
   */
  public void setActive(boolean active)
  {
    m_active = active;
		
    if (isVisible() && isPlaying())
      setMonitorEnabled(m_active);
  }
	
  /** 
   * Return is video controller active.
   */
  public boolean isActive()
  {
    return m_active;
  }
			
  /** 
   * Set the video controller as playing.
   */
  public void setPlaying(boolean playing)
  {
    m_playing = playing;
		
    setStreamEnabled(playing);	
		
    if (isActive() && isVisible())
      setMonitorEnabled(playing);
  }
	
  /** 
   * Return is video controller playing.
   */
  public boolean isPlaying()
  {
    return m_playing;
  }
	
  /** 
   * Set the video controller monitor state.
   */
  protected void setMonitorEnabled(boolean monitorEnabled)	
  {		
    if (monitorEnabled)		
      m_videoController.resumeMonitor();
    else m_videoController.pauseMonitor();
  }

  /** 
   * Set the video controller stream state.
   */
  protected void setStreamEnabled(boolean	streamEnabled)	
  {		
    if (streamEnabled)		
      m_videoController.resumeStream();	
    else m_videoController.pauseStream();	
  }
}
