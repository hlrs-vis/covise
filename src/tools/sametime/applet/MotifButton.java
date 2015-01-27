
import  java.awt.*;
import	java.awt.Component;
import	java.awt.CheckboxMenuItem;
import	java.awt.Panel;
import	java.awt.Checkbox;
import	java.awt.FlowLayout;
import	java.awt.Button;
import	java.awt.Menu;
import	java.awt.MenuItem;
import	java.awt.PopupMenu;

import	java.awt.event.ActionEvent;
import	java.awt.event.ActionListener;
import	java.awt.event.ItemEvent;
import	java.awt.event.ItemListener;


public class MotifButton extends Canvas  {

  String _text1, _text2;
  Image _i1, _i2;
  static Image _north, _south, _west, _east;  
  int _width, _height;
  int _thick;
  boolean _first;
  boolean _hasBorders;
  //private boolean _is_enabled;

  public MotifButton( Image i1, Image i2, String text1, String text2, int width ) {    
    _west  = MeetingLauncher.bwest;
    _south = MeetingLauncher.bsouth;
    _east  = MeetingLauncher.beast;
    _north = MeetingLauncher.bnorth;

    _i1 = i1;
    _i2 = i2;
    _text1 = text1; 
    _text2 = text2;   
    _width = width;    
    _height = 40;
    _thick=3;
    _first = false;
    _hasBorders = true;
    //_is_enabled = true;
    setSize( _width,_height);
    }

  public void setEnabled( boolean state) {
    //_is_enabled = state;
    this.setVisible(state);    
  }

  /*public boolean isEnabled() {
    return _is_enabled;
  }*/

  public void setState( boolean first) {
    _first = first;
  }

  public boolean getState() {
    return _first;
  }

  public void removeBorders() {
    _hasBorders = false;
  }

  public void paint( Graphics g) {      
      g.setColor( new Color(0x0));
      g.setFont( new Font("Dialog",Font.PLAIN,16) );
       
      if( _hasBorders ) {     
         g.drawImage( _north, 0, 0, _width-_thick, _thick,this );
         g.drawImage( _west, 0, _thick, _thick, _height-_thick,this );
         g.drawImage( _south, _thick-1, _height-_thick, _width-_thick+1, _thick,this );
         g.drawImage( _east,  _width-_thick, 0, _thick, _height-_thick,this );
      }

      if( _first ) {
         g.drawImage( _i1, _thick, _thick, _height-2*_thick-2, _height-2*_thick-2,this );  
         g.drawString(_text1, _height-_thick+2, _height-2*_thick-8);    
      }
      else {
         g.drawImage( _i2, _thick, _thick, _height-2*_thick-2, _height-2*_thick-2,this );  
         g.drawString(_text2, _height-_thick+2, _height-2*_thick-8);    
      }
   }
 }  
