/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* ADC initialisieren */
void ADC_Init(void)
{

    uint16_t result = 0;

    // interne Referenzspannung als Referenz für den ADC wählen:
    ADMUX = (1 << REFS0);

    // Bit ADFR ("free running") in ADCSRA steht beim Einschalten
    // schon auf 0, also single conversion
    ADCSRA = (1 << ADPS1) | (1 << ADPS0); // Frequenzvorteiler
    ADCSRA |= (1 << ADEN); // ADC aktivieren

    /* nach Aktivieren des ADC wird ein "Dummy-Readout" empfohlen, man liest
     also einen Wert und verwirft diesen, um den ADC "warmlaufen zu lassen" */

    ADCSRA |= (1 << ADSC); // eine ADC-Wandlung
    while (ADCSRA & (1 << ADSC))
    { // auf Abschluss der Konvertierung warten
    }
    /* ADCW muss einmal gelesen werden, sonst wird Ergebnis der nächsten
     Wandlung nicht übernommen. */
    result = ADCW;
}

//------------------------------------------------------------------------------

/* ADC Einzelmessung */
uint16_t ADC_Read(uint8_t channel)
{
    // Kanal waehlen, ohne andere Bits zu beeinflußen
    ADMUX = (ADMUX & ~(0x1F)) | (channel & 0x1F);
    ADCSRA |= (1 << ADSC); // eine Wandlung "single conversion"
    while (ADCSRA & (1 << ADSC))
    { // auf Abschluss der Konvertierung warten
    }
    return ADCW; // ADC auslesen und zurückgeben
}
// -----------------------------------------------------------------------------

/*

void adc_voltage(void)
{

	uint16_t ADCvalue;
	
	ADCvalue = ADC_Read(0);
	

	uint16_t ADCvalue_1;
	
	ADCvalue_1 = ADC_Read(1);
	

	if ((ADCvalue > initpositionxl) && (ADCvalue < initpositionxr)) {
		lcd_goto(0,12);
		lcd_writeText("  ",2);
	}
	if (ADCvalue < initpositionxl){
		lcd_goto(0,12);
		lcd_writeText("< ",2);
	}
	if (ADCvalue > initpositionxr){
		lcd_goto(0,12);
		lcd_writeText(" >",2);
	}
	
	if ((ADCvalue_1 > initpositionyd) && (ADCvalue_1 < initpositionyu)) {
		lcd_goto(0,14);
		lcd_writeText("  ",2);
	}
	if (ADCvalue_1 < initpositionyd){
		lcd_goto(0,14);
		lcd_writeText("v ",2);
	}
	if (ADCvalue_1 > initpositionyu){
		lcd_goto(0,14);
		lcd_writeText(" ^",2);
	}
}
*/