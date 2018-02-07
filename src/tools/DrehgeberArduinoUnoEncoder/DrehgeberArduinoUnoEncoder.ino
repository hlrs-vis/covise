#include <Encoder.h>

// Pin  Signal  Farbe  DrehgeberPin Farbe am Stecker innen
//  4     A0    gruen  3             grau
//  2     A1    gelb   5             grün
//  3     A2    orange 8             blau
//  5V    VCC   rot    2             Braun-schwarz
//  GND   GND   braun  11            weiss-schwarz

// Display zum Debugen
// SCL   SCL  braun    SCL
// SDA   SDA  grau     SDA
// 5V    VCC  weiss    VCC
// GND   GND  schwarz  GND

int portA0 =4; // D4 Reference signal
int portA1 =2; // D2
int portA2 =3; // D3
Encoder enc(portA1, portA2);
void setup() {
  
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(portA0, INPUT);
  digitalWrite(portA0,HIGH);
}
long encval=0;
long oldEncVal=0;
long writeCount = 0;
bool wasLarger = true;
int oldA0=1;
void loop()
{
    int a0 = digitalRead(portA0);
    encval = enc.read();
    if(writeCount < 1000000)
        writeCount++; 
    
    if(wasLarger == false && abs(encval)>20) // only send out a reference signal after we moved away from it (20 counts)
    {
      wasLarger = true;
    }
    if(a0 && a0!=oldA0 && wasLarger)
    {
      
      enc.write(0);
      wasLarger = false;
       Serial.println("reference");
       writeCount = 100000;
       
    }
    if(encval!=oldEncVal && writeCount > 1000) 
    {
       oldEncVal=encval;
       writeCount = 0; // don't write too often, only after a thousend iterations
       Serial.println(encval);
    
    }
    oldA0=a0;
}
