import java.awt.*;
import java.awt.event.*;

import com.lotus.sametime.streamedmedia.AudioController;

/** 
 * Audio Panel providing UI access to a single audio controller.
 */
public class AudioPanel extends Panel
{																	
  private AudioController m_audioController;
  private String m_audioLabelText;
  private Checkbox m_muteCheckbox;

  /** 
   * Audio panel constructor.
   */
  public AudioPanel(AudioController audioController, 
                    String audioLabelText)
  {
    m_audioController = audioController;
    m_audioLabelText = audioLabelText;
		
    layoutUI();
  }
	
  /** 
   * Layout the audio panel UI.
   */
  protected void layoutUI()
  {
    m_muteCheckbox = new Checkbox(m_audioLabelText, false);
    m_muteCheckbox.addItemListener(new MuteCheckboxHandler());
		
    setLayout(new BorderLayout());
    add("Center", m_muteCheckbox);
  }
	
  /** 
   * ItemListener implementation class to handle checkbox event.
   */
  class MuteCheckboxHandler implements ItemListener
  {
    public void itemStateChanged(ItemEvent event)
    {
      if (event.getStateChange() == ItemEvent.SELECTED)
        m_audioController.pauseStream();
      else m_audioController.resumeStream();
    }
  }
}
