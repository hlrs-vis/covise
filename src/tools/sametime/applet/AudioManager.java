import java.awt.*;
import java.awt.event.*;
import java.applet.*;

import com.lotus.sametime.core.comparch.*;
import com.lotus.sametime.places.Place;
import com.lotus.sametime.conference.*;
import com.lotus.sametime.streamedmedia.*;
import com.lotus.sametime.streamedmedia.event.*;

public class AudioManager  
{
  public static final String ACTIVITY_NAME = "Audio";
  public static final String SPEAKER_LABEL = "Mute Speaker";
  public static final String MICROPHONE_LABEL	= "Mute Microphone";
														
  private StreamedMediaInteractive m_multimediaController;
  private AudioController m_audioPlayer;
  private AudioController m_audioRecorder;

  private Panel m_audioPanel;
  private AudioPanel m_microphoneControl;
  private AudioPanel m_speakerControl;
	
  /** 
   * Connects the audio manager to the session and place.
   */
  public void connectAudio(STSession session, Place meetingPlace)
  {	
    StreamedMediaService streamedMediaService = 
      (StreamedMediaService)session.getCompApi
                  (StreamedMediaService.COMP_NAME);
    m_multimediaController = 
      streamedMediaService.getStreamedMediaForPlace(meetingPlace);
    try
    {	
      setupAudioDevices();				
      m_audioRecorder = m_multimediaController.getAudioRecorder();
      m_audioPlayer = m_multimediaController.getAudioPlayer();	
    }
    catch (Exception exception)
    {
      exception.printStackTrace();
    }			
    m_audioRecorder.resumeStream();	
    m_audioPlayer.resumeStream();
  }
  
  /** 
   * Setup the audio devices for work.
   */
  protected void setupAudioDevices()	
  {			
    try
    {			
      m_multimediaController.initAudio(null, null);								
    }
    catch (StreamedMediaException	exception)
    {
      exception.printStackTrace();
    }	
  }
  
  /**
   * Layout the audio manager panel.
   */
  public Panel layoutAudioUI()
  {
    m_audioPanel = new Panel();
    m_speakerControl = new AudioPanel(m_audioPlayer, 
                                      SPEAKER_LABEL);
    m_microphoneControl = new AudioPanel(m_audioRecorder, 
                                         MICROPHONE_LABEL);
		
    m_audioPanel.setLayout(new GridLayout(2, 1, 0, 0));
    m_audioPanel.add(m_speakerControl);
    m_audioPanel.add(m_microphoneControl);
		
    return m_audioPanel;
  }

   /** 
   * Start the audio devices.
   */
  public void startAudioDevices()
  {
    m_audioRecorder.resumeMonitor();
    m_audioPlayer.resumeMonitor();
  }
  
  /** 
   * Stop the audio devices.
   */
  public void stopAudioDevices()
  {
    m_audioRecorder.pauseMonitor();
    m_audioPlayer.pauseMonitor();
  }
}
