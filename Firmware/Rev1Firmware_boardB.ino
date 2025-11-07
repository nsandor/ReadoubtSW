/* ---------------------------------------------------------------
 * 10 × 10 switch/LED array (104‑bit frame) – Firmware v1.2
 *  Board‑ID splash (letters **A…E**), spiral reveal, master LED enable,
 *  verbose mode, ADC read on A7, 2‑byte SPI write, signal‑path GPIOs,
 *  measurement sequence command (**SEQ**), and reference‑source toggle (**REF**).
 *
 *  Target : Arduino Leonardo (ATmega32u4 @ 16 MHz)
 *  Serial : 9600 Bd  (USB‑CDC)
 *  SPI    : ICSP header (CS =D9)
 *
 *  Pin map
 *    DATA D7, CLOCK D5, OE̅ D6, CLR̅ D4
 *    LED_ENABLE D8
 *    TIA_SELECT A2 (D20), AMP_SELECT A1 (D19), ROUTE_OUTPUT A0 (D18)
 *    PSU CS D9
 *    ADC A7  (2.56 V ref)
 * ---------------------------------------------------------------- */

#include <Arduino.h>
#include <SPI.h>

/* ─── Firmware identifiers ─────────────────────────────────────── */
constexpr char    BOARD_ID       = 'B';   // compile‑time letter A‑E
constexpr uint8_t FW_VER_MAJOR   = 1;
constexpr uint8_t FW_VER_MINOR   = 2;

/* ─── Pin map ───────────────────────────────────────────────────── */
constexpr uint8_t PIN_DATA           = 7;
constexpr uint8_t PIN_CLOCK          = 5;
constexpr uint8_t PIN_OE             = 6;
constexpr uint8_t PIN_CLEAR          = 4;
constexpr uint8_t PIN_LED_ENABLE     = 8;

constexpr uint8_t PIN_SPI_CS         = 9;   // PSU CS

constexpr uint8_t PIN_OUTPUT_ROUTER  = A0;  // D18
constexpr uint8_t PIN_AMPLIFIER_SEL  = A1;  // D19
constexpr uint8_t PIN_TIA_SELECT     = A2;  // D20

constexpr uint8_t PIN_ADC            = A11;

/* ─── Constants ─────────────────────────────────────────────────── */
constexpr uint32_t SERIAL_BAUD   = 9600;
constexpr uint16_t FRAME_LEN     = 104;
constexpr uint16_t STEP_DELAY_MS = 10;
constexpr uint16_t INFO_MS       = 2000;
constexpr uint8_t  SWITCH_MIN    = 1;
constexpr uint8_t  SWITCH_MAX    = 100;

/* ─── Global state ─────────────────────────────────────────────── */
bool verbose          = true;
bool referenceActive  = false;   // tracks REF toggle state

/* ─── Splash bitmap (first 100 bits) ───────────────────────────── */
const uint8_t splash[10][10] PROGMEM = {
  {0,0,1,0,0,0,0,1,0,0},
  {0,0,1,1,0,0,1,1,0,0},
  {1,1,1,1,1,1,1,1,1,1},
  {0,1,1,1,0,0,1,1,1,0},
  {0,0,1,0,1,1,0,1,0,0},
  {0,0,1,0,1,1,0,1,0,0},
  {0,1,1,1,0,0,1,1,1,0},
  {1,1,1,1,1,1,1,1,1,1},
  {0,0,1,1,0,0,1,1,0,0},
  {0,0,1,0,0,0,0,1,0,0}
};
const char inactiveChannels[] = "1,2,3,4,6,7,8,9,10,11,12,13,14,21,22,23,24,35,38,47,48,55,56,57,58";

/* 5×7 glyphs for digits 0‑9 and letters A‑E (MSB‑left) ----------- */
const uint8_t FONT_DIGIT[10][7] PROGMEM = {
  {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, //0
  {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, //1
  {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, //2
  {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}, //3
  {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, //4
  {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, //5
  {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, //6
  {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, //7
  {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, //8
  {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}  //9
};
const uint8_t FONT_LET[5][7] PROGMEM = {          // A‑E
/*A*/{0x0E,0x11,0x11,0x1F,0x11,0x11,0x11},
/*B*/{0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E},
/*C*/{0x0E,0x11,0x10,0x10,0x10,0x11,0x0E},
/*D*/{0x1E,0x11,0x11,0x11,0x11,0x11,0x1E},
/*E*/{0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}
};

/* ─── Low‑level helpers ─────────────────────────────────────────── */
inline void pulseClock()         { digitalWrite(PIN_CLOCK, HIGH); digitalWrite(PIN_CLOCK, LOW); }
inline void shiftBit(bool v)     { digitalWrite(PIN_DATA,  v);     pulseClock(); }

void clearRegister(){
  digitalWrite(PIN_OE,HIGH);
  digitalWrite(PIN_CLEAR,LOW); delayMicroseconds(5);
  digitalWrite(PIN_CLEAR,HIGH);
}

void shiftFrameMSBFirst(const uint8_t *buf,uint16_t n){
  digitalWrite(PIN_OE,HIGH);
  for(int16_t i=n-1;i>=0;--i) shiftBit(buf[i]);
  digitalWrite(PIN_OE,LOW);
}

/* Build 104‑bit frame from internal index 0‑99 ------------------- */
void buildSwitchFrame(uint8_t idx,uint8_t *f){
  memset(f,0,FRAME_LEN);
  f[idx]=1;
  uint8_t a=idx/7;
  f[102]=(a>>2)&1; f[101]=(a>>1)&1; f[100]=a&1;
  f[103]=((idx/7)+1<9);
}

/* Build reference frame: 100 zeros + four 1s --------------------- */
void buildReferenceFrame(uint8_t *f){
  memset(f,0,FRAME_LEN);
  f[100]=f[101]=f[102]=f[103]=1;
}

String frameToString(const uint8_t *f){
  String s; s.reserve(FRAME_LEN);
  for(int16_t i=FRAME_LEN-1;i>=0;--i) s+=f[i]?'1':'0';
  return s;
}

/* ─── Spiral splash ------------------------------------------------ */
void makeSpiral(uint8_t ord[100]){
  bool used[10][10]={};
  const int8_t dr[4]={0,1,0,-1},dc[4]={1,0,-1,0};
  int r=4,c=4,d=0,step=1; uint8_t n=0;
  while(n<100){
    for(int k=0;k<step&&n<100;++k){
      if(r>=0&&r<10&&c>=0&&c<10&&!used[r][c]){
        ord[n++]=r*10+c; used[r][c]=true;
      }
      r+=dr[d]; c+=dc[d];
    }
    d=(d+1)&3; if(d==0||d==2) ++step;
  }
}
void spiralSplash(){
  static uint8_t path[100]; static bool ready=false;
  if(!ready){ makeSpiral(path); ready=true; }
  uint8_t frame[FRAME_LEN]={0};
  delay(1000);
  for(uint8_t i=0;i<100;++i){
    uint8_t p=path[i];
    if(i){ uint8_t q=path[i-1]; frame[q]=pgm_read_byte(&splash[q/10][q%10]); }
    frame[p]=1;
    shiftFrameMSBFirst(frame,FRAME_LEN);
    delay(STEP_DELAY_MS);
  }
  uint8_t last=path[99];
  frame[last]=pgm_read_byte(&splash[last/10][last%10]);
  shiftFrameMSBFirst(frame,FRAME_LEN);
  delay(500);
}

/* ─── Info splash ("<Letter><Digit>") ----------------------------- */
void buildInfoFrame(uint8_t f[FRAME_LEN])
{
  memset(f, 0, FRAME_LEN);

  /* pick letter glyph (A‑E) from PROGMEM */
  const uint8_t* glyphL = nullptr;
  if (BOARD_ID >= 'A' && BOARD_ID <= 'E')
    glyphL = FONT_LET[BOARD_ID - 'A'];

  /* copy digit glyph (0‑9) from PROGMEM into RAM */
  uint8_t glyphR[7];
  memcpy_P(glyphR, FONT_DIGIT[FW_VER_MAJOR % 10], 7);

  /* draw 5×7 glyphs into rows 1‑7 of the 10×10 frame */
  for (uint8_t row = 0; row < 7; ++row) {
    uint8_t L = glyphL ? pgm_read_byte(glyphL + row) : 0; // flash → RAM
    uint8_t R = glyphR[row];                              // already in RAM

    for (uint8_t col = 0; col < 5; ++col) {
      f[(row + 1) * 10 + col]       = (L >> (4 - col)) & 1;     // letter
      f[(row + 1) * 10 + 5 + col]   = (R >> (4 - col)) & 1;     // digit
    }
  }
}
void showInfoScreen(){
  uint8_t f[FRAME_LEN]; buildInfoFrame(f);
  shiftFrameMSBFirst(f,FRAME_LEN); delay(INFO_MS);
  clearRegister();
}

/* ─── ADC & SPI helpers ------------------------------------------ */
void printADC(){
  uint16_t raw=analogRead(PIN_ADC);
  float v=raw*2.56f/1023.0f;
  Serial.print(F("ADC raw=")); Serial.print(raw);
  Serial.print(F(" V=")); Serial.println(v,3);
}
void spiSend(uint8_t b1){
  digitalWrite(PIN_SPI_CS,LOW);
  SPI.transfer(b1);
  digitalWrite(PIN_SPI_CS,HIGH);
  Serial.print(F("SPI sent: 0x")); Serial.print(b1,HEX);
}

/* ─── Measurement sequence helper -------------------------------- */
void runMeasurementSequence(uint8_t start = SWITCH_MIN, uint8_t end = SWITCH_MAX,
                            uint16_t on_ms = STEP_DELAY_MS, uint16_t off_ms = STEP_DELAY_MS)
{
  /* sanity checks & normalisation */
  start = constrain(start, SWITCH_MIN, SWITCH_MAX);
  end   = constrain(end,   SWITCH_MIN, SWITCH_MAX);
  if (start > end) { uint8_t tmp = start; start = end; end = tmp; }

  uint8_t frame[FRAME_LEN];

  Serial.print(F("Running sequence ")); Serial.print(start); Serial.print('-'); Serial.print(end);
  Serial.print(F("  on=")); Serial.print(on_ms); Serial.print(F(" ms  off="));
  Serial.print(off_ms); Serial.println(F(" ms"));

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
bool isOn (const String&s){return s=="ON"||s=="1";}
bool isOff(const String&s){return s=="OFF"||s=="0";}
void setGpio(uint8_t pin,const String&v,const char*n){
  if(isOn(v)){digitalWrite(pin,HIGH); Serial.print(n); Serial.println(F(" → ON"));}
  else if(isOff(v)){digitalWrite(pin,LOW); Serial.print(n); Serial.println(F(" → OFF"));}
  else {Serial.print(n); Serial.println(F(": use ON/OFF or 1/0"));}
}

/* ─── Setup ------------------------------------------------------- */
void setup(){
  pinMode(PIN_DATA,OUTPUT); pinMode(PIN_CLOCK,OUTPUT);
  pinMode(PIN_OE,OUTPUT);   pinMode(PIN_CLEAR,OUTPUT);
  pinMode(PIN_LED_ENABLE,OUTPUT);
  pinMode(PIN_OUTPUT_ROUTER,OUTPUT);
  pinMode(PIN_AMPLIFIER_SEL,OUTPUT);
  pinMode(PIN_TIA_SELECT,OUTPUT);
  pinMode(PIN_SPI_CS,OUTPUT); digitalWrite(PIN_SPI_CS,HIGH);
  digitalWrite(PIN_LED_ENABLE,HIGH); clearRegister();
  digitalWrite(PIN_OUTPUT_ROUTER,HIGH);
  digitalWrite(PIN_AMPLIFIER_SEL,HIGH);
  analogReference(INTERNAL);           // 2.56 V
  SPI.begin();
  SPI.beginTransaction(SPISettings(125000,MSBFIRST,SPI_MODE0));

  Serial.begin(SERIAL_BAUD); while(!Serial){;}

  spiralSplash();
  showInfoScreen();
  digitalWrite(PIN_LED_ENABLE,LOW);
  Serial.print(F("FW ")); Serial.print(FW_VER_MAJOR); Serial.print('.');
  Serial.println(FW_VER_MINOR);
  Serial.print(F("Board ")); Serial.println(BOARD_ID);
  Serial.println(F("Ready – commands: 1‑100, SEQ, REF, LED, VERBOSE, ADC, SPI, TIA, AMP, ROUTE, IDN, SWSTATUS"));
}

/* ─── Loop -------------------------------------------------------- */
void loop(){
  if(!Serial.available()) return;
  String line=Serial.readStringUntil('\n'); line.trim();
  if(!line.length()) return;
  String up=line; up.toUpperCase();

  /* VERBOSE ------------------------------------------------------- */
  if(up.startsWith("VERBOSE")){
    up.remove(0,7); up.trim();
    if(isOn(up)){verbose=true; Serial.println(F("Verbose → ON")); }
    else if(isOff(up)){verbose=false; Serial.println(F("Verbose → OFF")); }
    else Serial.println(F("Use VERBOSE ON/OFF")); return;
  }

  /* LED ON/OFF ---------------------------------------------------- */
  if(up.startsWith("LED")){ up.remove(0,3); up.trim(); setGpio(PIN_LED_ENABLE,up,"LED"); return;}

  /* ADC read ------------------------------------------------------ */
  if(up=="ADC"||up=="READ A7"){ printADC(); return; }

  // Identifies board
  if(up=="IDN"){
    Serial.println("SaidaminovLab Readout Board RevA Board B");
    return;
  }
// Returns list of 
  if(up=="SWSTATUS"){
    Serial.println(inactiveChannels);
    return;
  }

  /* SPI byte write ------------------------------------------------ */
  if (up.startsWith("SPI")) {                 // e.g.  SPI 123
    up.remove(0, 3); up.trim();
    if (up.length() == 0) {
      Serial.println(F("SPI cmd: need one byte (0‑255)")); return;
    }
    long val = strtol(up.c_str(), nullptr, 10);
    if (val < 0 || val > 255) {
      Serial.println(F("SPI cmd: value out of range (0‑255)")); return;
    }
    uint8_t b = static_cast<uint8_t>(val);
    spiSend(b);

    Serial.print(F("  (bin: "));
    for (int i = 7; i >= 0; --i) Serial.print((b >> i) & 1);
    Serial.println(')');
    return;
  }

  /* SEQ  [start] [end] [on_ms] [off_ms] --------------------------- */
  if (up.startsWith("SEQ")) {
    up.remove(0,3); up.trim();

    uint8_t  start = SWITCH_MIN;
    uint8_t  end   = SWITCH_MAX;
    uint16_t on_ms = STEP_DELAY_MS;
    uint16_t off_ms= STEP_DELAY_MS;

    if (up.length()) {           // parse space‑separated parameters
      int params[4]; uint8_t n = 0;
      int last = 0;
      for (uint16_t i = 0; i <= up.length(); ++i) {
        if (i == up.length() || up[i] == ' ') {
          String tok = up.substring(last, i); tok.trim();
          if (tok.length() && n < 4) params[n++] = tok.toInt();
          last = i + 1;
        }
      }
      if (n > 0) start = params[0];
      if (n > 1) end   = params[1];
      if (n > 2) on_ms = params[2];
      if (n > 3) off_ms= params[3];
    }

    runMeasurementSequence(start, end, on_ms, off_ms);
    return;
  }

  /* REF toggle ---------------------------------------------------- */
  if (up=="REF"){                         // no parameters
    static uint8_t refFrame[FRAME_LEN];
    if(!referenceActive){
      buildReferenceFrame(refFrame);
      shiftFrameMSBFirst(refFrame, FRAME_LEN);
      referenceActive = true;
      Serial.println(F("Reference → ON"));
    }else{
      clearRegister();
      referenceActive = false;
      Serial.println(F("Reference → OFF"));
    }
    return;
  }

  /* GPIO helpers -------------------------------------------------- */
  if(up.startsWith("TIA"))  {up.remove(0,3); up.trim(); setGpio(PIN_TIA_SELECT,up,"TIA"); return;}
  if(up.startsWith("AMP"))  {up.remove(0,3); up.trim(); setGpio(PIN_AMPLIFIER_SEL,up,"AMP"); return;}
  if(up.startsWith("ROUTE")){up.remove(0,5); up.trim(); setGpio(PIN_OUTPUT_ROUTER,up,"ROUTE"); return;}

  /* Single switch (1‑100) ----------------------------------------- */
  long val=line.toInt();
  if(val>=SWITCH_MIN&&val<=SWITCH_MAX){
    uint8_t idx=val-1;
    uint8_t frame[FRAME_LEN]; buildSwitchFrame(idx,frame);
    shiftFrameMSBFirst(frame,FRAME_LEN);
    if(verbose){
      Serial.print(F("Switch ")); Serial.print(val); Serial.print(F(" → "));
      Serial.println(frameToString(frame));
    }else{
      Serial.println("ACK");
    }
  }else{
    Serial.println(F("Unknown cmd."));
  }
}
