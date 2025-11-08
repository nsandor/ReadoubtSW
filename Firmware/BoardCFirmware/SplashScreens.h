#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <EEPROM.h>
constexpr char BOARD_ID = 'D';  // compile-time letter A-E
constexpr uint8_t FW_VER_MAJOR = 2;
constexpr uint8_t FW_VER_MINOR = 1;
constexpr uint16_t FRAME_LENA = 104;
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

const uint8_t splashl[10][10] PROGMEM = {
  { 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
  { 0, 0, 1, 0, 0, 0, 1, 0, 1, 0 },
  { 0, 0, 1, 0, 0, 0, 1, 0, 1, 0 },
  { 0, 0, 1, 0, 0, 0, 1, 0, 1, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 1, 0, 1, 0, 1, 0, 0, 0, 0 },
  { 0, 1, 0, 1, 0, 1, 0, 0, 0, 0 },
  { 0, 1, 0, 1, 0, 1, 0, 0, 0, 0 },
  { 0, 1, 0, 1, 0, 1, 0, 1, 1, 1 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
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
/* ─── Info splash ("<Letter><Digit>") ----------------------------- */
void buildInfoFrame(uint8_t f[FRAME_LENA]) {
  memset(f, 0, FRAME_LENA);

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