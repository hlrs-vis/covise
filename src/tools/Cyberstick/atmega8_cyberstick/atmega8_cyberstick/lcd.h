/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ------------------------------------------------------
// ----------------------------------------------------------------------------
// lcd_send
// ----------------------------------------------------------------------------
// Sendet das Byte data unterteilt in jeweils 4 Bits an den 4 Bit Daten-
// port des LCD Displays. Nach schreiben der Daten auf den Bus muss jeweils
// E kurz eine Flanke setzen, um das Display zum Lesen zu bewegen. Normaler-
// weise müsste man jetzt das Display fragen, ob es die Daten fertig
// bearbeitet hat. Dazu muesste man in den Read-Mode und E auslesen?
// ----------------------------------------------------------------------------

#define sbi(port, bit) (port) |= (1 << (bit)) // set bit
#define cbi(port, bit) (port) &= ~(1 << (bit)) // clear bit

void lcd_send(char data)
{
    char status_d, tmp;

    cbi(PORTC, 2); // E = 0
    status_d = PORTD; // PortD auslesen
    status_d &= 0x0F; // LCD Pins nullen

    tmp = data; // data holen
    tmp &= 0xf0; // oberes Nibble ausmarkieren
    tmp |= status_d; // mit restlichen ursprungs-bits

    PORTD = tmp; // auf den PortD schreiben

    sbi(PORTC, 2); // Impuls auf E geben
    cbi(PORTC, 2);

    tmp = data << 4; // data holen und unteren Nibble <<
    tmp &= 0xf0; // oberes Nibble ausmarkieren
    tmp |= status_d; // mit restlichen ursprungs-bits

    PORTD = tmp; // auf den PortD schreiben
    sbi(PORTC, 2); // Impuls auf E geben
    cbi(PORTC, 2);

    _delay_ms(1); // dem Display etwas Zeit geben
}

// ----------------------------------------------------------------------------
// lcd_cmd
// ----------------------------------------------------------------------------
// Sendet ein Kommand an das LCD. Das Kommande wird an den Bus gelegt und
// RS am PortD2 wird low gesetzt, damit das LCD es als Kommando interpretiert.
// ----------------------------------------------------------------------------

void lcd_cmd(char cmd)
{
    cbi(PORTC, 3);
    lcd_send(cmd);
}

// ----------------------------------------------------------------------------
// lcd_clear
// ----------------------------------------------------------------------------
// Das Kommand 0x01 zum Löschen des LCD Displays wird per Kommando geschickt.
// ----------------------------------------------------------------------------

void lcd_clear(void)
{
    lcd_cmd(0x01);
    _delay_ms(2);
}

// ----------------------------------------------------------------------------
// lcd_home
// ----------------------------------------------------------------------------
// Die Methode setzt den Cursor des Displays wieder an die erste Stelle in
// der ersten Zeile.
// ----------------------------------------------------------------------------

void lcd_home(void)
{
    lcd_cmd(0x02);
    _delay_ms(2);
}

// ----------------------------------------------------------------------------
// lcd_on
// ----------------------------------------------------------------------------
// Sendet das Kommando an das LCD, um es einzuschalten.
// 0x0C Display ein / Cursor aus / kein Blinken
// ----------------------------------------------------------------------------

void lcd_on(void)
{
    lcd_cmd(0x0C);
}

// ----------------------------------------------------------------------------
// lcd_off
// ----------------------------------------------------------------------------
// Sende das Kommando 0x08 an das LCD, um es auszuschalten.
// ----------------------------------------------------------------------------

void lcd_off(void)
{
    lcd_cmd(0x08);
}

// ----------------------------------------------------------------------------
// lcd_goto
// ----------------------------------------------------------------------------
// Positioniert den Cursor an der Stelle row x col, damit dirt weiter-
// geschrieben werden kann.
// ----------------------------------------------------------------------------

void lcd_goto(int row, int col)
{
    unsigned char tmp = 0x00;

    switch (row)
    {
    case 0: // 0. Zeile
    {
        tmp = 0x80 + 0x00 + col;
        break;
    }
    case 1: // 1. Zeile
    {
        tmp = 0x80 + 0x40 + col;
        break;
    }
    case 2: // 2. Zeile
    {
        tmp = 0x80 + 0x10 + col;
        break;
    }
    case 3: // 3. Zeile
    {
        tmp = 0x80 + 0x50 + col;
        break;
    }
    }

    lcd_cmd(tmp);
}

// ----------------------------------------------------------------------------
// lcd_init
// ----------------------------------------------------------------------------
// Die Methode setzt die Ausgaenge fuer das Displaymodul entsprechend.
// Die Befehle zum initilaisieren des Displays werden gesendet und dann
// das Display neu gestart.
// ----------------------------------------------------------------------------

void lcd_init(void)
{
    DDRD = 0xFC; // PortD2-7 als Ausgang setzen
    PORTD &= 0x03; // PortD2-7 auf 0
    DDRC |= (1 << PC2) | (1 << PC3); // PortC0-1 als Ausgang setzen
    PORTC &= ~((1 << PC2) | (1 << PC3)); // PortC0-1 auf 0
    _delay_ms(50);

    PORTC &= ~((1 << PC2) | (1 << PC3)); // PortC0-1 weider auf 0s
    PORTD &= 0x03; // PortD2-7 wieder auf 0s
    sbi(PORTD, 5); // 0x001000xy senden (4Bit-Mode setzen)

    sbi(PORTC, 2); // Impuls auf E geben
    cbi(PORTC, 2);
    _delay_ms(5);

    lcd_off(); // ???
    _delay_ms(5);

    lcd_cmd(0x28); // Kommando: 4Bit / 2 Zeilen / 5x7
    lcd_off(); // lcd ausschalten
    lcd_clear(); // display loeschen
    lcd_cmd(0x06); // Kommando: Inkrement / kein Scrollen
    lcd_on(); // display einschalten
}

// ----------------------------------------------------------------------------
// lcd_write
// ----------------------------------------------------------------------------
// Die Methode schreibt an die aktuelle Position des Cursors das ihr
// uebergebene Zeichen.
// ----------------------------------------------------------------------------

void lcd_write(char text)
{
    sbi(PORTC, 3); // RS setzen
    lcd_send(text); // Zeichen schicken
}

// ----------------------------------------------------------------------------
// lcd_writeText
// ----------------------------------------------------------------------------
// Sendet die Anzahl size im uebergebenen char-Array enthaltenen Zeichen an
// das Display.
// ----------------------------------------------------------------------------

void lcd_writeText(char *pText, int size)
{
    int i;

    for (i = 0; i < size; ++i)
    {
        lcd_write(pText[i]);
    }
}
//------------------------------------------------------------------------------