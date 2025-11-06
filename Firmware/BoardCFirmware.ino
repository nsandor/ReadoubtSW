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

/* ─── Firmware identifiers ─────────────────────────────────────── */
constexpr char BOARD_ID = 'C';  // compile-time letter A-E
constexpr uint8_t FW_VER_MAJOR = 2;
constexpr uint8_t FW_VER_MINOR = 1;

/* ─── Pin map ───────────────────────────────────────────────────── */
constexpr uint8_t PIN_DATA = 7;
constexpr uint8_t PIN_CLOCK = 5;
constexpr uint8_t PIN_OE = 6;
constexpr uint8_t PIN_CLEAR = 4;
constexpr uint8_t PIN_LED_ENABLE = 8;
constexpr uint8_t PIN_SWINT_HOLD = 9;
constexpr uint8_t PIN_SWINT_RESET = 10;
constexpr uint8_t PIN_ADCDAT = 11;
constexpr uint8_t PIN_SPI_CS = A5;  // PSU CS

constexpr uint8_t PIN_OUTPUT_ROUTER = A0;  // D18
constexpr uint8_t PIN_AMPLIFIER_SEL = A1;  // D19
constexpr uint8_t PIN_TIA_SELECT = A2;     // D20
constexpr uint8_t PIN_SWINV_SELECT = A3;   // D20

constexpr uint8_t PIN_ADC = A7;

/* ─── General constants ────────────────────────────────────────── */
constexpr uint32_t SERIAL_BAUD = 9600;
constexpr uint16_t FRAME_LEN = 104;
constexpr uint16_t STEP_DELAY_MS = 10;
constexpr uint16_t INFO_MS = 2000;
constexpr uint8_t SWITCH_MIN = 1;
constexpr uint8_t SWITCH_MAX = 101;

/* ─── External ADC (ADS112C04) constants ─────────────────────────
 * Address assumes A1=A0 strapped low → 0b1000000 = 0x40.
 *
 * Command bytes:
 *   RESET      0x06
 *   START/SYNC 0x08
 *   POWERDOWN  0x02
 *   RDATA      0x10
 *   RREG       0x20 | (reg<<2)
 *   WREG       0x40 | (reg<<2)
 *
 * We configure:
 *   - Differential AIN0(+) - AIN1(-)
 *   - Gain = 1
 *   - PGA bypassed
 *   - Single-shot mode, 20 SPS
 *   - Internal 2.048 V ref
 *
 * Full-scale is ±2.048 V at gain=1.
 * LSB ~= 4.096 V / 65536 counts.
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

/* ─── Splash bitmap (first 100 bits) ───────────────────────────── */
const uint8_t splash[10][10] PROGMEM = {
  { 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 },
  { 0, 0, 1, 1, 0, 0, 1, 1, 0, 0 },
  { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
  { 0, 1, 1, 1, 0, 0, 1, 1, 1, 0 },
  { 0, 0, 1, 0, 1, 1, 0, 1, 0, 0 },
  { 0, 0, 1, 0, 1, 1, 0, 1, 0, 0 },
  { 0, 1, 1, 1, 0, 0, 1, 1, 1, 0 },
  { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
  { 0, 0, 1, 1, 0, 0, 1, 1, 0, 0 },
  { 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 }
};

/* 5×7 glyphs for digits 0-9 and letters A-E (MSB-left) ----------- */
const uint8_t FONT_DIGIT[10][7] PROGMEM = {
  { 0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E },  //0
  { 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E },  //1
  { 0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F },  //2
  { 0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E },  //3
  { 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02 },  //4
  { 0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E },  //5
  { 0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E },  //6
  { 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08 },  //7
  { 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E },  //8
  { 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C }   //9
};

const uint8_t FONT_LET[5][7] PROGMEM = {  // A-E
  /*A*/ { 0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 },
  /*B*/ { 0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E },
  /*C*/ { 0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E },
  /*D*/ { 0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E },
  /*E*/ { 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F }
};

/* ─── Low-level helpers (shift reg / LED matrix) ───────────────── */
inline void pulseClock() {
  digitalWrite(PIN_CLOCK, HIGH);
  digitalWrite(PIN_CLOCK, LOW);
}
inline void shiftBit(bool v) {
  digitalWrite(PIN_DATA, v);
  pulseClock();
}

void clearRegister() {
  digitalWrite(PIN_OE, HIGH);
  digitalWrite(PIN_CLEAR, LOW);
  delayMicroseconds(5);
  digitalWrite(PIN_CLEAR, HIGH);
}

void shiftFrameMSBFirst(const uint8_t *buf, uint16_t n) {
  digitalWrite(PIN_OE, HIGH);
  for (int16_t i = n - 1; i >= 0; --i) shiftBit(buf[i]);
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

/* ─── Spiral splash ------------------------------------------------ */
void makeSpiral(uint8_t ord[100]) {
  bool used[10][10] = {};
  const int8_t dr[4] = { 0, 1, 0, -1 }, dc[4] = { 1, 0, -1, 0 };
  int r = 4, c = 4, d = 0, step = 1;
  uint8_t n = 0;
  while (n < 100) {
    for (int k = 0; k < step && n < 100; ++k) {
      if (r >= 0 && r < 10 && c >= 0 && c < 10 && !used[r][c]) {
        ord[n++] = r * 10 + c;
        used[r][c] = true;
      }
      r += dr[d];
      c += dc[d];
    }
    d = (d + 1) & 3;
    if (d == 0 || d == 2) ++step;
  }
}

void spiralSplash() {
  static uint8_t path[100];
  static bool ready = false;
  if (!ready) {
    makeSpiral(path);
    ready = true;
  }

  uint8_t frame[FRAME_LEN] = { 0 };
  delay(1000);

  for (uint8_t i = 0; i < 100; ++i) {
    uint8_t p = path[i];
    if (i) {
      uint8_t q = path[i - 1];
      frame[q] = pgm_read_byte(&splash[q / 10][q % 10]);
    }
    frame[p] = 1;
    shiftFrameMSBFirst(frame, FRAME_LEN);
    delay(STEP_DELAY_MS);
  }

  uint8_t last = path[99];
  frame[last] = pgm_read_byte(&splash[last / 10][last % 10]);
  shiftFrameMSBFirst(frame, FRAME_LEN);
  delay(500);
}

/* ─── Info splash ("<Letter><Digit>") ----------------------------- */
void buildInfoFrame(uint8_t f[FRAME_LEN]) {
  memset(f, 0, FRAME_LEN);

  /* pick letter glyph (A-E) from PROGMEM */
  const uint8_t *glyphL = nullptr;
  if (BOARD_ID >= 'A' && BOARD_ID <= 'E')
    glyphL = FONT_LET[BOARD_ID - 'A'];

  /* copy digit glyph (0-9) from PROGMEM into RAM */
  uint8_t glyphR[7];
  memcpy_P(glyphR, FONT_DIGIT[FW_VER_MAJOR % 10], 7);

  /* draw 5×7 glyphs into rows 1-7 of the 10×10 frame */
  for (uint8_t row = 0; row < 7; ++row) {
    uint8_t L = glyphL ? pgm_read_byte(glyphL + row) : 0;  // flash → RAM
    uint8_t R = glyphR[row];                               // already in RAM

    for (uint8_t col = 0; col < 5; ++col) {
      f[(row + 1) * 10 + col] = (L >> (4 - col)) & 1;      // letter
      f[(row + 1) * 10 + 5 + col] = (R >> (4 - col)) & 1;  // digit
    }
  }
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
}

/* ─── On-chip ADC helper (A7, unipolar) --------------------------- */
void printADC_local() {
  uint16_t raw = analogRead(PIN_ADC);
  float v = raw * 2.56f / 1023.0f;
  Serial.print(F("ADC raw="));
  Serial.print(raw);
  Serial.print(F(" V="));
  Serial.println(v, 3);
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
 *   - MUX = AIN0(+) - AIN1(-)
 *   - GAIN = 1
 *   - PGA_BYPASS = 1
 *   - Single-shot mode, 20 SPS
 *   - Internal 2.048V reference
 *
 * REG0 bits:
 *   [7:4] MUX[3:0]  = 0000  (AIN0 vs AIN1 differential)
 *   [3:1] GAIN[2:0] = 000   (gain = 1)
 *   [0]   PGA_BYPASS= 1     (bypass PGA)
 *   -> 0000 0001 = 0x01
 *
 * REG1 bits:
 *   DR[2:0]=000 (20SPS)
 *   MODE=0 (normal)
 *   CM=0 (single-shot)
 *   VREF[1:0]=00 (internal 2.048V)
 *   TS=0 (normal operation)
 *   -> 0x00
 *
 * REG2, REG3 left at 0x00 (defaults).
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
 *   1. START/SYNC command (0x08) to trigger conversion in single-shot mode
 *   2. wait ~60 ms (20 SPS → ~50 ms conv time, we add margin)
 *   3. send RDATA (0x10) via write, then do repeated-start read of 2 bytes
 *
 * Returns true on success, fills rawCode (signed 16-bit).
 * rawCode is two's complement, full-scale ±2.048 V at gain=1.
 */
bool ads112c04_singleShotReadRaw(int16_t &rawCode) {
  ads112c04_sendCommand(ADS112C04_CMD_STARTSYNC);
  while (digitalRead(PIN_ADCDAT) == HIGH) {}
  //delay(60); // blocking wait; DRDY pin not used

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
 * Print ADC2 result:
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

void runSwitchedIntegrator(uint16_t integration_time) {
  //Measurement seq should...
  // Pull hold high
  digitalWrite(PIN_SWINT_HOLD, HIGH);
  delayMicroseconds(10);
  // Pull reset low
  digitalWrite(PIN_SWINT_RESET, LOW);
  delayMicroseconds(10);
  // Pull reset high again
  digitalWrite(PIN_SWINT_RESET, HIGH);
  delayMicroseconds(10);
  //Then pull hold low and hold it there for the rest of the integration time
  digitalWrite(PIN_SWINT_HOLD, LOW);
  delay(10);
  //delayMicroseconds(integration_time-30);
  digitalWrite(PIN_SWINT_HOLD, HIGH);
  int16_t raw;
  if (!ads112c04_singleShotReadRaw(raw)) {
    Serial.println(F("ADC2 error (no data)"));
    return;
  }
  // raw is two's complement, full-scale ±2.048 V at gain=1.
  // LSB ≈ 62.5 µV.
  float zeroed = raw - 16499;
  float volts = zeroed * ADS112C04_LSB_VOLTS;
  //float nanoAmps = -1*volts/20000000*1000000000;
  Serial.print(F("ADC2 raw="));
  Serial.print(raw);
  Serial.print(F(" V="));
  Serial.println(volts, 6);
  digitalWrite(PIN_SWINT_HOLD, LOW);
  //Serial.print(F(" nA="));
  //Serial.println(nanoAmps,6);
}

void setVoltage(uint16_t voltage) {
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
  pinMode(PIN_SWINT_HOLD, OUTPUT);
  pinMode(PIN_SWINT_RESET, OUTPUT);
  pinMode(PIN_SPI_CS, OUTPUT);
  digitalWrite(PIN_SPI_CS, HIGH);
  digitalWrite(PIN_SWINT_HOLD, LOW);
  digitalWrite(PIN_SWINT_RESET, HIGH);
  // Enable LEDs by default, blank shift regs
  digitalWrite(PIN_LED_ENABLE, HIGH);
  clearRegister();

  // Startup visuals
  spiralSplash();
  showInfoScreen();

  // Analog ref for on-chip ADC
  analogReference(INTERNAL);  // 2.56 V internal ref for A7

  // SPI init for PSU control
  SPI.begin();
  SPI.beginTransaction(SPISettings(125000, MSBFIRST, SPI_MODE0));

  // I2C init for ADS112C04 external ADC
  Wire.begin();
  ads112c04_init();

  // Serial init
  Serial.begin(SERIAL_BAUD);
  while (!Serial) { ; }  // wait for USB CDC

  Serial.print(F("FW "));
  Serial.print(FW_VER_MAJOR);
  Serial.print('.');
  Serial.println(FW_VER_MINOR);

  Serial.print(F("Board "));
  Serial.println(BOARD_ID);

  Serial.println(F("Ready – commands:"));
  Serial.println(F("  1-100        : drive individual switch"));
  Serial.println(F("  REF          : drive only address bits (100/101/102=1)"));
  Serial.println(F("  SEQ ...      : run measurement sequence"));
  Serial.println(F("  LED ON/OFF   : master LED enable"));
  Serial.println(F("  VERBOSE ON/OFF"));
  Serial.println(F("  ADC          : read on-chip A7 (unipolar)"));
  Serial.println(F("  ADC2         : read ext. ADS112C04 (bipolar diff AIN0-AIN1)"));
  Serial.println(F("  SPI <0-255>  : write byte to PSU over SPI"));
  Serial.println(F("  TIA/AMP/ROUTE ON|OFF"));
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

  /* ADC (on-chip A7, unipolar) ----------------------------------- */
  if (up == "ADC" || up == "READ A7") {
    printADC_local();
    return;
  }

  /* ADC2 (external ADS112C04 over I2C, bipolar diff) -------------- */
  if (up == "ADC2") {
    printADC_ads112c04();
    return;
  }

  if (up == "SWINTR") {
    runSwitchedIntegrator(100000);
    return;
  }

  /* SPI byte write ------------------------------------------------ */
  if (up.startsWith("SPI")) {  // e.g.  SPI 123
    up.remove(0, 3);
    up.trim();
    if (up.length() == 0) {
      Serial.println(F("SPI cmd: need one byte (0-255)"));
      return;
    }
    long ival = strtol(up.c_str(), nullptr, 10);
    if (ival < 0 || ival > 255) {
      Serial.println(F("SPI cmd: value out of range (0-255)"));
      return;
    }
    uint8_t b = static_cast<uint8_t>(ival);
    spiSend(b);

    Serial.print(F("  (bin: "));
    for (int i = 7; i >= 0; --i) Serial.print((b >> i) & 1);
    Serial.println(')');
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
  if (up.startsWith("MEASLOCAL")) {
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
      float nanoAmps = -1 * volts / 20000000 * 1000000000;
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
      Serial.print(F("Switch "));
      Serial.println(val);
    }
    return;
  }

  /* Fallback ------------------------------------------------------ */
  Serial.println(F("Unknown cmd."));
}
