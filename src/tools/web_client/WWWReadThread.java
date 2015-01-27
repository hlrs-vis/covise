import java.awt.*;

import java.lang.StringBuffer;
import java.net.*;
import java.io.*;



class WWWReadThread extends Thread
{
  private static WWWConnection www_conn;
  private static BufferedReader in;
  private static UserInterface	user_interface;

  WWWReadThread()
  {
     WWWConnection.read_thread = this;
  }
 
  public void init(WWWConnection conn, BufferedReader i) {
     this.user_interface = ClientApplet.user_interface;
     this.www_conn = conn;
     this.in = i;
  }

  public void run()
  { 
     String msg;
     boolean exit;
 
     exit = false;
     while ((www_conn.read_thread == this)&&(!exit))
     {
        msg = read_HTTP_msg();
        if(msg==null) 
        {
           System.err.println("\n!!!! ReadThread null msg !!!\n");
           exit = true;
        }
        else 
        {
           www_conn.handle_wwwmsg(msg);
           msg = null;
           //Runtime.getRuntime().gc();
        }  
            
     }
     System.err.println("\n!!!! ReadThread exiting !!!\n"); 
  }

  String read_HTTP_msg()
  {
     String inputLine;
     int length;
     boolean end;

     length = 0;
     end = false;    
     try {
        while ( !end)
        {
           inputLine = in.readLine();
           if(inputLine == null) 
           {
              System.err.println("\n!!!! ReadThread - inputLine = null  !!!\n"); 
              break;
           }
           if(inputLine.length() == 0) 
           {
              end = true;  // end of headers
              //user_interface.addInfo("\n!!!! ReadThread - end of headers  !!!\n"); 
           }
           else
           {
              if(inputLine.startsWith("Content-Length:"))
              {
                 Integer l = Integer.valueOf(inputLine.substring(15));
                 length = l.intValue();
                 //user_interface.addInfo("\n!!!! ReadThread -length =");
                 //user_interface.addInfo(length);
                 //user_interface.addInfo("!!!\n"); 
              } 
           }
        }
        if(length != 0) // reading the content
        {
           char data[];
           try {
              data = new char[length+1];
           } catch (OutOfMemoryError e) {
               user_interface.printInfo("ERROR : " +e+ " -> please reload the applet !!!");
               www_conn.disconnect();
               return null; 
           }
           
           int tot = 0;
           int res = 0;
           while((tot < length) && (res>=0))
           {   
              res = in.read(data,tot,(length-tot));
              if(res>0) tot += res; 
              //user_interface.addInfo("\n!!!! ReadThread -");
              //user_interface.addInfo(res);
              //user_interface.addInfo("bytes read !!!");
           } 
           //user_interface.addInfo("\n!!!! ReadThread - EOF !!!\n");
           data[length] = '\0';
           String content = new String(data);
           //inputLine = in.readLine(); 
           return content;
        } 
     } catch (IOException e) {
        System.err.println("!!!! ReadThread -  read_HTTP_msg failed " + e);
        System.exit(1);
     }
     return null;
  } 
 



}



