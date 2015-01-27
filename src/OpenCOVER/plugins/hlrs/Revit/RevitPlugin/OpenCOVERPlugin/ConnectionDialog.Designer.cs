namespace OpenCOVERPlugin
{
   partial class ConnectionDialog
   {
      /// <summary>
      /// Required designer variable.
      /// </summary>
      private System.ComponentModel.IContainer components = null;

      /// <summary>
      /// Clean up any resources being used.
      /// </summary>
      /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
      protected override void Dispose(bool disposing)
      {
         if (disposing && (components != null))
         {
            components.Dispose();
         }
         base.Dispose(disposing);
      }

      #region Windows Form Designer generated code

      /// <summary>
      /// Required method for Designer support - do not modify
      /// the contents of this method with the code editor.
      /// </summary>
      private void InitializeComponent()
      {
         this.HostnameLabel = new System.Windows.Forms.Label();
         this.hostnameText = new System.Windows.Forms.TextBox();
         this.label1 = new System.Windows.Forms.Label();
         this.portText = new System.Windows.Forms.TextBox();
         this.connectButton = new System.Windows.Forms.Button();
         this.cancelButton = new System.Windows.Forms.Button();
         this.SuspendLayout();
         // 
         // HostnameLabel
         // 
         this.HostnameLabel.AutoSize = true;
         this.HostnameLabel.Location = new System.Drawing.Point(30, 9);
         this.HostnameLabel.Name = "HostnameLabel";
         this.HostnameLabel.Size = new System.Drawing.Size(55, 13);
         this.HostnameLabel.TabIndex = 0;
         this.HostnameLabel.Text = "Hostname";
         // 
         // hostnameText
         // 
         this.hostnameText.Location = new System.Drawing.Point(33, 27);
         this.hostnameText.Name = "hostnameText";
         this.hostnameText.Size = new System.Drawing.Size(206, 20);
         this.hostnameText.TabIndex = 1;
         this.hostnameText.Text = "localhost";
         // 
         // label1
         // 
         this.label1.AutoSize = true;
         this.label1.Location = new System.Drawing.Point(30, 60);
         this.label1.Name = "label1";
         this.label1.Size = new System.Drawing.Size(26, 13);
         this.label1.TabIndex = 2;
         this.label1.Text = "Port";
         // 
         // portText
         // 
         this.portText.Location = new System.Drawing.Point(33, 76);
         this.portText.Name = "portText";
         this.portText.Size = new System.Drawing.Size(52, 20);
         this.portText.TabIndex = 3;
         this.portText.Text = "31821";
         // 
         // connectButton
         // 
         this.connectButton.Location = new System.Drawing.Point(33, 111);
         this.connectButton.Name = "connectButton";
         this.connectButton.Size = new System.Drawing.Size(75, 23);
         this.connectButton.TabIndex = 4;
         this.connectButton.Text = "Connect";
         this.connectButton.UseVisualStyleBackColor = true;
         this.connectButton.Click += new System.EventHandler(this.connectButton_Click);
         // 
         // cancelButton
         // 
         this.cancelButton.DialogResult = System.Windows.Forms.DialogResult.Cancel;
         this.cancelButton.Location = new System.Drawing.Point(164, 111);
         this.cancelButton.Name = "cancelButton";
         this.cancelButton.Size = new System.Drawing.Size(75, 23);
         this.cancelButton.TabIndex = 5;
         this.cancelButton.Text = "Cancel";
         this.cancelButton.UseVisualStyleBackColor = true;
         this.cancelButton.Click += new System.EventHandler(this.cancelButton_Click);
         // 
         // ConnectionDialog
         // 
         this.AcceptButton = this.connectButton;
         this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.CancelButton = this.cancelButton;
         this.ClientSize = new System.Drawing.Size(286, 146);
         this.ControlBox = false;
         this.Controls.Add(this.cancelButton);
         this.Controls.Add(this.connectButton);
         this.Controls.Add(this.portText);
         this.Controls.Add(this.label1);
         this.Controls.Add(this.hostnameText);
         this.Controls.Add(this.HostnameLabel);
         this.Name = "ConnectionDialog";
         this.Text = "ConnectionDialog";
         this.ResumeLayout(false);
         this.PerformLayout();

      }

      #endregion

      private System.Windows.Forms.Label HostnameLabel;
      private System.Windows.Forms.TextBox hostnameText;
      private System.Windows.Forms.Label label1;
      private System.Windows.Forms.TextBox portText;
      private System.Windows.Forms.Button connectButton;
      private System.Windows.Forms.Button cancelButton;
   }
}