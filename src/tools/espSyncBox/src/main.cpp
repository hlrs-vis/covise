#include <Arduino.h>
#include <Preferences.h>

// Both outputs carry the same signal. NORMAL is a 50% duty square wave;
// PULSE is a pulse train with a configurable high time.
constexpr uint8_t OUTPUT_PINS[] = {32, 33};
constexpr uint32_t DEFAULT_FREQUENCY_HZ = 60;
constexpr uint32_t MIN_FREQUENCY_HZ = 1;
constexpr uint32_t MAX_FREQUENCY_HZ = 1000;
constexpr uint32_t DEFAULT_PULSE_WIDTH_US = 1000;
constexpr uint32_t MIN_PULSE_WIDTH_US = 1;
constexpr char PREF_NAMESPACE[] = "espSyncBox";

// Value 2 was the old standalone invert mode; retain it for NVS migration.
enum class OutputMode : uint8_t { Normal = 0, Pulse = 1, Vga = 3, VgaNegative = 4 };

Preferences preferences;
OutputMode mode = OutputMode::Normal;
bool invertSignal = false;
uint32_t pulseWidthUs = DEFAULT_PULSE_WIDTH_US;
uint32_t frequencyHz = DEFAULT_FREQUENCY_HZ;
uint32_t cycleStartUs = 0;
bool outputHigh = false;
String commandBuffer;

uint32_t periodUs() {
    return 1000000UL / frequencyHz;
}

uint32_t maxPulseWidthUs() {
    return periodUs() - 1;
}

void setOutputs(bool high) {
    if (high == outputHigh) return;
    outputHigh = high;
    for (uint8_t pin : OUTPUT_PINS) digitalWrite(pin, high ? HIGH : LOW);
}

void saveSettings() {
    preferences.putUChar("mode", static_cast<uint8_t>(mode));
    preferences.putBool("invert", invertSignal);
    preferences.putUInt("width_us", pulseWidthUs);
    preferences.putUInt("freq_hz", frequencyHz);
}

void printStatus() {
    const char *modeName = mode == OutputMode::Pulse ? "pulse" :
                           mode == OutputMode::Vga ? "vga" :
                           mode == OutputMode::VgaNegative ? "vga-" : "normal";
    Serial.printf("Mode: %s, invert: %s, pulse width: %lu us, frequency: %lu Hz\n",
                  modeName, invertSignal ? "on" : "off",
                  static_cast<unsigned long>(pulseWidthUs), static_cast<unsigned long>(frequencyHz));
}

void printHelp() {
    Serial.println("Commands:");
    Serial.println("  MODE normal       Square wave, 50% duty");
    Serial.println("  MODE pulse        Pulse train");
    Serial.println("  MODE vga          Positive VGA vertical sync pulse");
    Serial.println("  MODE vga-         Negative VGA vertical sync pulse");
    Serial.println("  MODE invert       Compatibility alias for INVERT on");
    Serial.println("  INVERT on|off     Invert every output signal");
    Serial.println("  WIDTH <us>        Set sync pulse width (1..period-1 microseconds)");
    Serial.println("  FREQ <hz>         Set frequency (1..1000 Hz)");
    Serial.println("  STATUS            Show current settings");
    Serial.println("  HELP              Show this help");
}

void processCommand(String line) {
    line.trim();
    if (line.isEmpty()) return;

    int separator = line.indexOf(' ');
    String verb = (separator < 0 ? line : line.substring(0, separator));
    String argument = separator < 0 ? "" : line.substring(separator + 1);
    verb.toUpperCase();
    argument.trim();

    if (verb == "MODE") {
        argument.toLowerCase();
        if (argument == "normal") mode = OutputMode::Normal;
        else if (argument == "pulse") mode = OutputMode::Pulse;
        else if (argument == "vga") mode = OutputMode::Vga;
        else if (argument == "vga-") mode = OutputMode::VgaNegative;
        else if (argument == "invert") {
            invertSignal = true;
            saveSettings();
            Serial.println("OK: invert=on (current waveform unchanged)");
            return;
        }
        else {
            Serial.println("ERR: use MODE normal, pulse, vga, vga-, or invert");
            return;
        }
        saveSettings();
        cycleStartUs = micros();
        Serial.printf("OK: mode=%s\n", mode == OutputMode::Pulse ? "pulse" : mode == OutputMode::Vga ? "vga" : mode == OutputMode::VgaNegative ? "vga-" : "normal");
    } else if (verb == "INVERT") {
        argument.toLowerCase();
        if (argument == "on" || argument == "1") invertSignal = true;
        else if (argument == "off" || argument == "0") invertSignal = false;
        else {
            Serial.println("ERR: use INVERT on or INVERT off");
            return;
        }
        saveSettings();
        Serial.printf("OK: invert=%s\n", invertSignal ? "on" : "off");
    } else if (verb == "WIDTH") {
        if (argument.isEmpty()) {
            Serial.println("ERR: use WIDTH <microseconds>");
            return;
        }
        char *end = nullptr;
        unsigned long parsed = strtoul(argument.c_str(), &end, 10);
        if (end == argument.c_str() || *end != '\0' || parsed < MIN_PULSE_WIDTH_US || parsed > maxPulseWidthUs()) {
            Serial.printf("ERR: width must be %lu..%lu microseconds at the current frequency\n",
                          static_cast<unsigned long>(MIN_PULSE_WIDTH_US),
                          static_cast<unsigned long>(maxPulseWidthUs()));
            return;
        }
        pulseWidthUs = static_cast<uint32_t>(parsed);
        saveSettings();
        Serial.printf("OK: width_us=%lu\n", static_cast<unsigned long>(pulseWidthUs));
    } else if (verb == "FREQ") {
        if (argument.isEmpty()) {
            Serial.println("ERR: use FREQ <hertz>");
            return;
        }
        char *end = nullptr;
        unsigned long parsed = strtoul(argument.c_str(), &end, 10);
        if (end == argument.c_str() || *end != '\0' || parsed < MIN_FREQUENCY_HZ || parsed > MAX_FREQUENCY_HZ) {
            Serial.printf("ERR: frequency must be %lu..%lu Hz\n",
                          static_cast<unsigned long>(MIN_FREQUENCY_HZ),
                          static_cast<unsigned long>(MAX_FREQUENCY_HZ));
            return;
        }
        frequencyHz = static_cast<uint32_t>(parsed);
        if (pulseWidthUs > maxPulseWidthUs()) pulseWidthUs = maxPulseWidthUs();
        saveSettings();
        cycleStartUs = micros();
        Serial.printf("OK: frequency=%lu Hz, width_us=%lu\n",
                      static_cast<unsigned long>(frequencyHz), static_cast<unsigned long>(pulseWidthUs));
    } else if (verb == "STATUS") {
        printStatus();
    } else if (verb == "HELP") {
        printHelp();
    } else {
        Serial.println("ERR: unknown command; send HELP");
    }
}

void readSerialCommands() {
    while (Serial.available()) {
        char c = static_cast<char>(Serial.read());
        if (c == '\r' || c == '\n') {
            processCommand(commandBuffer);
            commandBuffer = "";
        } else if (commandBuffer.length() < 80) {
            commandBuffer += c;
        } else {
            commandBuffer = "";
            Serial.println("ERR: command too long");
        }
    }
}

void updateWaveform() {
    const uint32_t now = micros();
    const uint32_t currentPeriodUs = periodUs();
    uint32_t elapsed = now - cycleStartUs; // unsigned subtraction handles micros() wraparound
    if (elapsed >= currentPeriodUs) {
        cycleStartUs += (elapsed / currentPeriodUs) * currentPeriodUs;
        elapsed = now - cycleStartUs;
    }

    bool signalHigh;
    if (mode == OutputMode::Normal) {
        signalHigh = elapsed < currentPeriodUs / 2;
    } else if (mode == OutputMode::VgaNegative) {
        signalHigh = elapsed >= pulseWidthUs;
    } else {
        signalHigh = elapsed < pulseWidthUs;
    }
    setOutputs(invertSignal ? !signalHigh : signalHigh);
}

void setup() {
    Serial.begin(115200);
    for (uint8_t pin : OUTPUT_PINS) {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, LOW);
    }

    preferences.begin(PREF_NAMESPACE, false);
    uint8_t storedMode = preferences.getUChar("mode", static_cast<uint8_t>(OutputMode::Normal));
    if (storedMode == 0 || storedMode == 1 || storedMode == 3 || storedMode == 4) {
        mode = static_cast<OutputMode>(storedMode);
    } else {
        mode = OutputMode::Normal;
    }
    invertSignal = preferences.getBool("invert", storedMode == 2);
    frequencyHz = preferences.getUInt("freq_hz", DEFAULT_FREQUENCY_HZ);
    if (frequencyHz < MIN_FREQUENCY_HZ || frequencyHz > MAX_FREQUENCY_HZ) {
        frequencyHz = DEFAULT_FREQUENCY_HZ;
        preferences.putUInt("freq_hz", frequencyHz);
    }
    pulseWidthUs = preferences.getUInt("width_us", DEFAULT_PULSE_WIDTH_US);
    if (pulseWidthUs < MIN_PULSE_WIDTH_US || pulseWidthUs > maxPulseWidthUs()) {
        pulseWidthUs = min(DEFAULT_PULSE_WIDTH_US, maxPulseWidthUs());
        preferences.putUInt("width_us", pulseWidthUs);
    }

    cycleStartUs = micros();
    Serial.println("ESP32 sync output ready. Send HELP for commands.");
    printStatus();
}

void loop() {
    readSerialCommands();
    updateWaveform();
}
