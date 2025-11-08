/* ---------------------------------------------------------------
 * 10×10 switch/LED array (104-bit frame) – Firmware v1.2
 *
 * Features:
 *  - Board-ID splash (letters A…E)
 *  - Spiral reveal animation
 *  - Master LED enable (LED command)
 *  - VERBOSE mode for serial output
 *  - On-chip ADC read on A7 (ADC / READ A7)
 *  - External ADS112C04 I2C ADC read (ADC2), bipolar differential
 *  - SPI 1-byte write for PSU control (SPI <byte>)
 *  - Signal-path GPIOs (TIA, AMP, ROUTE)
 *  - Measurement sequence command (SEQ)
 *  - REF command: drives only bits 100/101/102 high
 *
 * Target : Arduino Leonardo (ATmega32u4 @ 16 MHz)
 * Serial : 9600 Bd  (USB-CDC)
 * SPI    : ICSP header (CS = A5)
 * I2C    : default SDA/SCL pins
 *
 * Pin map
 *    DATA D7, CLOCK D5, OE̅ D6, CLR̅ D4
 *    LED_ENABLE D8
 *    TIA_SELECT A2 (D20), AMP_SELECT A1 (D19), ROUTE_OUTPUT A0 (D18)
 *    PSU CS A5
 *    On-chip ADC input: A7  (2.56 V ref)
 *
 * ADS112C04 external ADC:
 *    Connected on I2C (SDA/SCL). DRDY is NOT wired; we poll.
 *    Configured for differential AIN0(+) - AIN1(−), bipolar about 0 V.
 * ---------------------------------------------------------------- */

#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <EEPROM.h>
#include "SplashScreens.h"


/* ─── Pin map ───────────────────────────────────────────────────── */
constexpr uint8_t PIN_DATA = 7;
constexpr uint8_t PIN_CLOCK = 5;
constexpr uint8_t PIN_OE = 6;
constexpr uint8_t PIN_CLEAR = 4;
constexpr uint8_t PIN_LED_ENABLE = 8;
constexpr uint8_t PIN_ADCDAT = 11;
constexpr uint8_t PIN_SPI_CS = A5;  // PSU CS
constexpr uint8_t PIN_OUTPUT_ROUTER = A0;  // D18
constexpr uint8_t PIN_AMPLIFIER_SEL = A1;  // D19
constexpr uint8_t PIN_TIA_SELECT = A2;     // D20
constexpr uint8_t PIN_SWINV_SELECT = A3;   // D20
/* ─── General constants ────────────────────────────────────────── */
constexpr uint32_t SERIAL_BAUD = 9600;
constexpr uint16_t FRAME_LEN = 104;
constexpr uint16_t STEP_DELAY_MS = 10;
constexpr uint16_t INFO_MS = 2000;
constexpr uint8_t SWITCH_MIN = 1;
constexpr uint8_t SWITCH_MAX = 101;

/* ─── External ADC (ADS112C04) constants ─────────────────────────
 */
constexpr uint8_t ADS112C04_ADDR = 0x40;
constexpr uint8_t ADS112C04_CMD_RESET = 0x06;
constexpr uint8_t ADS112C04_CMD_STARTSYNC = 0x08;
constexpr uint8_t ADS112C04_CMD_POWERDOWN = 0x02;
constexpr uint8_t ADS112C04_CMD_RDATA = 0x10;
// 4.096 V span = (+2.048 to -2.048), bipolar about 0.
constexpr float ADS112C04_LSB_VOLTS = 0.000142;

/* ─── Global state ─────────────────────────────────────────────── */
bool verbose = true;
bool local = false;
float calValue;
/* ─── Low-level helpers (shift reg / LED matrix) ───────────────── */

void clearRegister() {
  digitalWrite(PIN_OE, HIGH);
  digitalWrite(PIN_CLEAR, LOW);
  delayMicroseconds(5);
  digitalWrite(PIN_CLEAR, HIGH);
}

void shiftFrameMSBFirst(const uint8_t *buf, uint16_t n) {
  digitalWrite(PIN_OE, HIGH);
  for (int16_t i = n - 1; i >= 0; --i){
    digitalWrite(PIN_DATA, buf[i]);
    // Clock pulse
    digitalWrite(PIN_CLOCK, HIGH);
    digitalWrite(PIN_CLOCK, LOW);
  }
  digitalWrite(PIN_OE, LOW);
}

/*
 * Build 104-bit frame for a given switch index (0-99).
 *
 * Layout:
 *   f[idx] = 1 selects that switch.
 *   f[100],f[101],f[102] encode idx/7 (3-bit address).
 *   f[103] = 1 if ((idx/7)+1 < 9).
 */
void buildSwitchFrame(uint8_t idx, uint8_t *f) {
  memset(f, 0, FRAME_LEN);
  f[idx] = 1;
  uint8_t a = idx / 7;
  f[102] = (a >> 2) & 1;
  f[101] = (a >> 1) & 1;
  f[100] = a & 1;
  f[103] = ((idx / 7) + 1 < 9);
}

/*
 * Build special REF frame:
 * All bits 0 except 100,101,102 = 1. Bit 103 = 0.
 */
void buildRefFrame(uint8_t *f) {
  memset(f, 0, FRAME_LEN);
  f[100] = 1;
  f[101] = 1;
  f[102] = 1;
  // f[103] left 0
}

/*
 * Convert frame[0..103] to a MSB-first string for debug.
 * Bit 103 printed first, bit 0 last.
 */
String frameToString(const uint8_t *f) {
  String s;
  s.reserve(FRAME_LEN);
  for (int16_t i = FRAME_LEN - 1; i >= 0; --i) s += f[i] ? '1' : '0';
  return s;
}

void spiralSplash() {
  static uint8_t path[100];
  makeSpiral(path);

  uint8_t frame[FRAME_LEN] = { 0 };
  long randomval = random(100);

  for (uint8_t i = 0; i < 100; ++i) {
    uint8_t p = path[i];
    if (i) {
      uint8_t q = path[i - 1];
      if (randomval>90){
        frame[q] = pgm_read_byte(&splashl[q / 10][q % 10]);
      }
      else{
        frame[q] = pgm_read_byte(&splash[q / 10][q % 10]);
      }
      
    }
    frame[p] = 1;
    shiftFrameMSBFirst(frame, FRAME_LEN);
    delay(STEP_DELAY_MS);
  }

  uint8_t last = path[99];
  if (randomval>90){
  frame[last] = pgm_read_byte(&splashl[last / 10][last % 10]);
  }
  else{
  frame[last] = pgm_read_byte(&splash[last / 10][last % 10]);
  }
  shiftFrameMSBFirst(frame, FRAME_LEN);
  delay(500);
}

void showInfoScreen() {
  uint8_t f[FRAME_LEN];
  buildInfoFrame(f);
  shiftFrameMSBFirst(f, FRAME_LEN);
  delay(INFO_MS);
  clearRegister();
}

/* ─── SPI helper -------------------------------------------------- */
void spiSend(uint8_t b1) {
  digitalWrite(PIN_SPI_CS, LOW);
  SPI.transfer(b1);
  digitalWrite(PIN_SPI_CS, HIGH);
  Serial.print(F("SPI sent: 0x"));
  Serial.print(b1, HEX);
  Serial.println();
}

/* ─── ADS112C04 low-level I2C helpers ----------------------------- */
void ads112c04_sendCommand(uint8_t cmd) {
  Wire.beginTransmission(ADS112C04_ADDR);
  Wire.write(cmd);
  Wire.endTransmission();  // STOP
}

void ads112c04_writeRegister(uint8_t regAddr, uint8_t value) {
  Wire.beginTransmission(ADS112C04_ADDR);
  Wire.write(0x40 | (regAddr << 2));  // WREG opcode
  Wire.write(value);
  Wire.endTransmission();  // STOP
}

uint8_t ads112c04_readRegister(uint8_t regAddr) {
  // send RREG command, then repeated-start read 1 byte
  Wire.beginTransmission(ADS112C04_ADDR);
  Wire.write(0x20 | (regAddr << 2));  // RREG opcode
  Wire.endTransmission(false);        // repeated START after this

  Wire.requestFrom(ADS112C04_ADDR, (uint8_t)1);
  if (Wire.available() < 1) return 0xFF;
  return Wire.read();
}

/*
 * Configure ADS112C04 for bipolar differential measurement:
 */
void ads112c04_init() {
  // Wait >500 µs after power-up before I2C traffic
  delayMicroseconds(1000);
  // Force known state
  ads112c04_sendCommand(ADS112C04_CMD_RESET);
  delayMicroseconds(50);
  //uint8_t reg0 = 0b10100001; // differential AIN2-AVSS, gain=1, PGA bypass, SWINT use
  uint8_t reg0 = 0x81;  // differential AIN0-AIN1, gain=1, PGA bypass, TIA use
  //uint8_t reg1 = 0x04; // 20SPS, single-shot, external rails ref, normal mode
  uint8_t reg1 = 0b11010100;  // differential AIN2-AVSS, gain=1, PGA bypass, SWINT use
  uint8_t reg2 = 0x00;        // default
  uint8_t reg3 = 0x00;        // default
  ads112c04_writeRegister(0, reg0);
  ads112c04_writeRegister(1, reg1);
  ads112c04_writeRegister(2, reg2);
  ads112c04_writeRegister(3, reg3);
}

/*
 * Take one single-shot reading:
 */
bool ads112c04_singleShotReadRaw(int16_t &rawCode) {
  ads112c04_sendCommand(ADS112C04_CMD_STARTSYNC);
  while (digitalRead(PIN_ADCDAT) == HIGH) {}

  // send RDATA to request result
  Wire.beginTransmission(ADS112C04_ADDR);
  Wire.write(ADS112C04_CMD_RDATA);
  Wire.endTransmission(false);  // repeated-start for read

  Wire.requestFrom(ADS112C04_ADDR, (uint8_t)2);
  if (Wire.available() < 2) {
    return false;
  }
  uint8_t hi = Wire.read();
  uint8_t lo = Wire.read();
  rawCode = (int16_t)((hi << 8) | lo);
  return true;
}

/*
 * Print ADC result:
 *   raw signed code and converted volts (bipolar, differential).
 */
void printADC_ads112c04() {
  int16_t raw;
  if (!ads112c04_singleShotReadRaw(raw)) {
    Serial.println(F("ADC2 error (no data)"));
    return;
  }

  // raw is two's complement, full-scale ±2.048 V at gain=1.
  // LSB ≈ 62.5 µV.
  float zeroed = raw - 16499;
  float volts = zeroed * ADS112C04_LSB_VOLTS;
  float nanoAmps = -1 * volts / 20000000 * 1000000000;
  Serial.print(F("ADC2 raw="));
  Serial.print(raw);
  Serial.print(F(" V="));
  Serial.println(volts, 6);
  Serial.print(F(" nA="));
  Serial.println(nanoAmps, 6);
}

/* ─── Measurement sequence helper --------------------------------
 * SEQ [start] [end] [on_ms] [off_ms]
 * Iterates switches start→end, enabling each for on_ms,
 * clearing, then waiting off_ms.
 */
void runMeasurementSequence(uint8_t start = SWITCH_MIN,
                            uint8_t end = SWITCH_MAX,
                            uint16_t on_ms = STEP_DELAY_MS,
                            uint16_t off_ms = STEP_DELAY_MS) {
  start = constrain(start, SWITCH_MIN, SWITCH_MAX);
  end = constrain(end, SWITCH_MIN, SWITCH_MAX);
  if (start > end) {
    uint8_t tmp = start;
    start = end;
    end = tmp;
  }

  uint8_t frame[FRAME_LEN];
  Serial.print(F("Running sequence "));
  Serial.print(start);
  Serial.print('-');
  Serial.print(end);
  Serial.print(F("  on="));
  Serial.print(on_ms);
  Serial.print(F(" ms  off="));
  Serial.print(off_ms);
  Serial.println(F(" ms"));

  for (uint8_t val = start; val <= end; ++val) {
    buildSwitchFrame(val - 1, frame);
    shiftFrameMSBFirst(frame, FRAME_LEN);
    delay(on_ms);

    clearRegister();
    delay(off_ms);
  }
  Serial.println(F("Sequence done."));
}

/* ─── Util -------------------------------------------------------- */
bool isOn(const String &s) {
  return s == "ON" || s == "1";
}
bool isOff(const String &s) {
  return s == "OFF" || s == "0";
}

void setGpio(uint8_t pin, const String &v, const char *n) {
  if (isOn(v)) {
    digitalWrite(pin, HIGH);
    Serial.print(n);
    Serial.println(F(" → ON"));
  } else if (isOff(v)) {
    digitalWrite(pin, LOW);
    Serial.print(n);
    Serial.println(F(" → OFF"));
  } else {
    Serial.print(n);
    Serial.println(F(": use ON/OFF or 1/0"));
  }
}

/* ─── Setup ------------------------------------------------------- */
void setup() {
  // GPIO directions
  pinMode(PIN_DATA, OUTPUT);
  pinMode(PIN_CLOCK, OUTPUT);
  pinMode(PIN_ADCDAT, INPUT);
  pinMode(PIN_OE, OUTPUT);
  pinMode(PIN_CLEAR, OUTPUT);
  pinMode(PIN_LED_ENABLE, OUTPUT);
  pinMode(PIN_OUTPUT_ROUTER, OUTPUT);
  pinMode(PIN_AMPLIFIER_SEL, OUTPUT);
  pinMode(PIN_TIA_SELECT, OUTPUT);
  pinMode(PIN_SWINV_SELECT, OUTPUT);
  pinMode(PIN_SPI_CS, OUTPUT);
  digitalWrite(PIN_SPI_CS, HIGH);
  digitalWrite(PIN_LED_ENABLE, HIGH);
  randomSeed(EEPROM.read(0));
  //EEPROM.write(0,random(1000));
  calValue = EEPROM.read(10);
  clearRegister();
  // Startup visuals
  spiralSplash();
  showInfoScreen();
  
  // SPI init for PSU control
  SPI.begin();
  SPI.beginTransaction(SPISettings(125000, MSBFIRST, SPI_MODE0));

  // I2C init for ADS112C04 external ADC
  Wire.begin();
  ads112c04_init();

  // Serial init
  Serial.begin(SERIAL_BAUD);
  while (!Serial) { ; }  // wait for USB CDC
  Serial.println(F("SaidaminovLab Switching Readout Board"));
  Serial.print(F("FW "));
  Serial.print(FW_VER_MAJOR);
  Serial.print('.');
  Serial.println(FW_VER_MINOR);

  Serial.print(F("Board "));
  Serial.println(BOARD_ID);

  Serial.print(F("Calibration value: "));
  Serial.println(calValue);

  Serial.println(F("Ready – commands:"));
  Serial.println(F("  1-100        : drive individual switch"));
  Serial.println(F("  IDN          : identification string"));
  Serial.println(F("  SWSTATUS     : get switch status"));
  Serial.println(F("  REF          : Route reference 5ish nA source to output"));
  Serial.println(F("  SEQ ...      : run measurement sequence"));
  Serial.println(F("  LED ON/OFF   : master LED enable"));
  Serial.println(F("  VERBOSE ON/OFF"));
  Serial.println(F("  ADC         : read ext. ADS112C04 (bipolar diff AIN0-AIN1)"));
  Serial.println(F("  SETVOLT <Voltage>  : Set Local Voltage Output (6-87V)"));
  Serial.println(F("  MEASURE_LOCAL   : Measure all inputs using local readout"));
  Serial.println(F("  MEASURE_EXTERNAL   : Configure for external readout"));
  Serial.println(F("  TIA/AMP/ROUTE ON|OFF : Output flow control (refer to docs)"));
}

/* ─── Loop -------------------------------------------------------- */
void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (!line.length()) return;

  String up = line;
  up.toUpperCase();

  /* VERBOSE ------------------------------------------------------- */
  if (up.startsWith("VERBOSE")) {
    up.remove(0, 7);
    up.trim();
    if (isOn(up)) {
      verbose = true;
      Serial.println(F("Verbose → ON"));
    } else if (isOff(up)) {
      verbose = false;
      Serial.println(F("Verbose → OFF"));
    } else {
      Serial.println(F("Use VERBOSE ON/OFF"));
    }
    return;
  }

  /* LED ON/OFF ---------------------------------------------------- */
  if (up.startsWith("LED")) {
    up.remove(0, 3);
    up.trim();
    setGpio(PIN_LED_ENABLE, up, "LED");
    return;
  }

  /* REF frame (drive only bits 100/101/102 = 1) ------------------- */
  if (up == "REF") {
    uint8_t frame[FRAME_LEN];
    buildRefFrame(frame);
    shiftFrameMSBFirst(frame, FRAME_LEN);
    if (verbose) {
      Serial.println(F("REF frame sent (100,101,102 HIGH; others LOW):"));
      Serial.println(frameToString(frame));
    } else {
      Serial.println(F("REF frame sent"));
    }
    return;
  }


  /* ADC2 (external ADS112C04 over I2C, bipolar diff) -------------- */
  if (up == "ADC") {
    printADC_ads112c04();
    return;
  }

  if (up.startsWith("SETVOLT")){
    up.remove(0,7);
    up.trim();
    if (up.length()==0){
      Serial.println(F("Voltage Set Command: 6-87V"));
      return;
    }
    long voltage_value = strtol(up.c_str(),nullptr,10);
    if (voltage_value<6 || voltage_value>87){
      Serial.println(F("Voltage out of range (6-87V)"));
      return;
    }
    // Round voltage to nearest possible value given 256 steps
    float step_size = (87.6 - 5.72) / 255.0;
    float nearest_voltage = round((voltage_value - 5.72) / step_size) * step_size + 6.0;
    // Spi 1 = max, 255 = min
    uint8_t spi_value = static_cast<uint8_t>(255 - ((nearest_voltage - 5.72) / step_size));
    spiSend(spi_value);
    Serial.print(F("Voltage set to: "));
    Serial.print(nearest_voltage,2);
    Serial.println(F(" V"));
    return;
  }

  /* SEQ  [start] [end] [on_ms] [off_ms] --------------------------- */
  if (up.startsWith("SEQ")) {
    up.remove(0, 3);
    up.trim();

    uint8_t start = SWITCH_MIN;
    uint8_t end = SWITCH_MAX;
    uint16_t on_ms = STEP_DELAY_MS;
    uint16_t off_ms = STEP_DELAY_MS;

    if (up.length()) {
      int params[4];
      uint8_t n = 0;
      int last = 0;
      for (uint16_t i = 0; i <= up.length(); ++i) {
        if (i == up.length() || up[i] == ' ') {
          String tok = up.substring(last, i);
          tok.trim();
          if (tok.length() && n < 4) {
            params[n++] = tok.toInt();
          }
          last = i + 1;
        }
      }
      if (n > 0) start = params[0];
      if (n > 1) end = params[1];
      if (n > 2) on_ms = params[2];
      if (n > 3) off_ms = params[3];
    }

    runMeasurementSequence(start, end, on_ms, off_ms);
    return;
  }

  /* GPIO helpers -------------------------------------------------- */
  if (up.startsWith("TIA")) {
    up.remove(0, 3);
    up.trim();
    setGpio(PIN_TIA_SELECT, up, "TIA");
    return;
  }

  if (up.startsWith("SWINV")) {
    up.remove(0, 5);
    up.trim();
    setGpio(PIN_SWINV_SELECT, up, "SWINV");
    return;
  }

  if(up=="IDN"){
    Serial.println("SaidaminovLab Readout Board RevB Board C");
    return;
  }

  if(up=="SWSTATUS"){
    //Serial.println(inactiveChannels);
    return;
  }

  if (up.startsWith("AMP")) {
    up.remove(0, 3);
    up.trim();
    setGpio(PIN_AMPLIFIER_SEL, up, "AMP");
    return;
  }

  if (up.startsWith("ROUTE")) {
    up.remove(0, 5);
    up.trim();
    setGpio(PIN_OUTPUT_ROUTER, up, "ROUTE");
    return;
  }

  if (up.startsWith("SETCAL")){
    up.remove(0, 3);
    up.trim();
    float calValue = up.toFloat();
    EEPROM.write(10,calValue);
    Serial.print(F("Calibration value set to: "));
    Serial.println(calValue);
    //setVoltage(static_cast<uint16_t>(calValue));
    return;
  }

  if (up.startsWith("MEASURE_EXTERNAL")){
    if (local){
      setGpio(PIN_AMPLIFIER_SEL, "OFF", "AMP"); 
      setGpio(PIN_TIA_SELECT, "OFF", "TIA");
      setGpio(PIN_OUTPUT_ROUTER, "ON", "ROUTE");
      local = false;
    } 
    return;
  }

  if (up.startsWith("MEASURE_LOCAL")) {
    if (!local){
      setGpio(PIN_AMPLIFIER_SEL, "ON", "AMP"); 
      setGpio(PIN_TIA_SELECT, "ON", "TIA");
      setGpio(PIN_OUTPUT_ROUTER, "OFF", "ROUTE");
      local = true;
    }
    unsigned long starttime = millis();
    int16_t ADCVALS[100];
    for (uint8_t idx = 0; idx <= 99; idx++) {
      uint8_t frame[FRAME_LEN];
      buildSwitchFrame(idx, frame);
      shiftFrameMSBFirst(frame, FRAME_LEN);
      delay(1); //Settling time
      ads112c04_singleShotReadRaw(ADCVALS[idx]);
    }
    unsigned long endtime = millis();

    for (uint8_t idx = 0; idx <= 99; idx++) {
      float zeroed = ADCVALS[idx] - 16499;
      float volts = zeroed * ADS112C04_LSB_VOLTS;
      float nanoAmps = calValue*-1 * volts / 20000000 * 1000000000;
      Serial.println(nanoAmps);
    }
    unsigned long timeelapsed = endtime - starttime;
    Serial.println(timeelapsed);
    return;
  }

  /* Single switch (1-100) ----------------------------------------- */
  long val = line.toInt();
  if (val >= SWITCH_MIN && val <= SWITCH_MAX) {
    uint8_t idx = val - 1;
    uint8_t frame[FRAME_LEN];
    buildSwitchFrame(idx, frame);
    shiftFrameMSBFirst(frame, FRAME_LEN);

    if (verbose) {
      Serial.print(F("Switch "));
      Serial.print(val);
      Serial.print(F(" → "));
      Serial.println(frameToString(frame));
    } else {
      Serial.println("ACK");
    }
    return;
  }

  /* Fallback ------------------------------------------------------ */
  Serial.println(F("Unknown cmd."));
}
