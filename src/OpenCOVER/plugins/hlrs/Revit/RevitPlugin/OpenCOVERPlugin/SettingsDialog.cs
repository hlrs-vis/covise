using Autodesk.Revit.DB;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using Form = System.Windows.Forms.Form;

namespace OpenCOVERPlugin
{
   public partial class SettingsDialog : Form
   {
      private Document doc;
      public SettingsDialog(Document document)
        {
            InitializeComponent();
            connectState = false;
            this.doc = document;
        }
        public string getHostname()
      {
         return CAVEhost.Text;
      }

        public void setHost(String s)
        { CAVEhost.Text = s; }
        public void setDoRotate(bool b)
        { doRotate.Checked = b; }   

        public void setCoordinateSystem(COVER.CoordSystem coordSystem)
        {
            CoordianteSystem.SelectedIndex = ((int)coordSystem-1);
        }
        public COVER.CoordSystem getCoordSystem()
        { return (COVER.CoordSystem)CoordianteSystem.SelectedIndex; }
        public bool getDoRotate()
        { return doRotate.Checked; }
        public String getHost()
        { return CAVEhost.Text; }

        private CheckBox doRotate;
        private Label label1;
        private ComboBox CoordianteSystem;
        private Label label2;
        private TextBox CAVEhost;
        private Button Accept;
        private Button Cancel;
        private bool connectState;
     
        private void InitializeComponent()
        {
            this.doRotate = new System.Windows.Forms.CheckBox();
            this.label1 = new System.Windows.Forms.Label();
            this.CoordianteSystem = new System.Windows.Forms.ComboBox();
            this.label2 = new System.Windows.Forms.Label();
            this.CAVEhost = new System.Windows.Forms.TextBox();
            this.Accept = new System.Windows.Forms.Button();
            this.Cancel = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // doRotate
            // 
            this.doRotate.AutoSize = true;
            this.doRotate.Location = new System.Drawing.Point(12, 12);
            this.doRotate.Name = "doRotate";
            this.doRotate.Size = new System.Drawing.Size(116, 29);
            this.doRotate.TabIndex = 0;
            this.doRotate.Text = "doRotate";
            this.doRotate.UseVisualStyleBackColor = true;
            this.doRotate.CheckedChanged += new System.EventHandler(this.checkBox1_CheckedChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 44);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(174, 25);
            this.label1.TabIndex = 1;
            this.label1.Text = "CoordianteSystem";
            // 
            // CoordianteSystem
            // 
            this.CoordianteSystem.FormattingEnabled = true;
            this.CoordianteSystem.Items.AddRange(new object[] {
            "Origin",
            "Project Base Point",
            "Shared Base Point",
            "Geo Referenced"});
            this.CoordianteSystem.Location = new System.Drawing.Point(12, 81);
            this.CoordianteSystem.Name = "CoordianteSystem";
            this.CoordianteSystem.Size = new System.Drawing.Size(235, 32);
            this.CoordianteSystem.TabIndex = 2;
            this.CoordianteSystem.Tag = "";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(12, 142);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(110, 25);
            this.label2.TabIndex = 3;
            this.label2.Text = "CAVE host";
            this.label2.Click += new System.EventHandler(this.label2_Click);
            // 
            // CAVEhost
            // 
            this.CAVEhost.Location = new System.Drawing.Point(12, 184);
            this.CAVEhost.Name = "CAVEhost";
            this.CAVEhost.Size = new System.Drawing.Size(235, 29);
            this.CAVEhost.TabIndex = 4;
            this.CAVEhost.Text = "visent.hlrs.de";
            // 
            // Accept
            // 
            this.Accept.Location = new System.Drawing.Point(65, 252);
            this.Accept.Name = "Accept";
            this.Accept.Size = new System.Drawing.Size(95, 34);
            this.Accept.TabIndex = 5;
            this.Accept.Text = "Accept";
            this.Accept.UseVisualStyleBackColor = true;
            this.Accept.Click += new System.EventHandler(this.Accept_Click);
            // 
            // Cancel
            // 
            this.Cancel.Location = new System.Drawing.Point(194, 252);
            this.Cancel.Name = "Cancel";
            this.Cancel.Size = new System.Drawing.Size(95, 34);
            this.Cancel.TabIndex = 6;
            this.Cancel.Text = "Cancel";
            this.Cancel.UseVisualStyleBackColor = true;
            this.Cancel.Click += new System.EventHandler(this.Cancel_Click);
            // 
            // SettingsDialog
            // 
            this.ClientSize = new System.Drawing.Size(367, 312);
            this.Controls.Add(this.Cancel);
            this.Controls.Add(this.Accept);
            this.Controls.Add(this.CAVEhost);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.CoordianteSystem);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.doRotate);
            this.Name = "SettingsDialog";
            this.Load += new System.EventHandler(this.SettingsDialog_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        private void SettingsDialog_Load(object sender, EventArgs e)
        {

        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void Cancel_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void Accept_Click(object sender, EventArgs e)
        {
            COVER.Instance.doRotate = doRotate.Checked;
            COVER.Instance.currentCoordSystem = (COVER.CoordSystem)(CoordianteSystem.SelectedIndex+1);
            COVER.Instance.CAVEHost = CAVEhost.Text;
            COVER.Instance.storeConnectionInfo(doc);
            Close();
        }
    }
}