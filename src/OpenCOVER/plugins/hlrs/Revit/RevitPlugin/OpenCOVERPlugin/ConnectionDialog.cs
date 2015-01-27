using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace OpenCOVERPlugin
{
   public partial class ConnectionDialog : Form
   {
      public ConnectionDialog()
      {
         InitializeComponent();
         connectState = false;
      }
      public string getHostname()
      {
         return hostnameText.Text;
      }

      public int getPort()
      {
         return Convert.ToInt32(portText.Text);
      }
      private bool connectState;
      public bool connect()
      {
         return connectState;
      }
      private void connectButton_Click(object sender, EventArgs e)
      {
         connectState = true;
         Close();
      }

      private void cancelButton_Click(object sender, EventArgs e)
      {

         connectState = false;
         Close();
      }
   }
}