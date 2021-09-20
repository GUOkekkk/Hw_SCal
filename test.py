# include <LiquidCrystal.h>
// LCD
library

LiquidCrystal
lcd(7, 6, 5, 4, 3, 2);
volatile
int
setpoint = 100;
volatile
int
setpoint_stop = setpoint;
unsigned
long
deadline = 100;
const
int
OFF = 0;
const
int
PUSH = 1;
const
int
ON = 2;
const
int
RELEASE = 3;
// Set
the
state

void
setup()
{
    pinMode(A0, INPUT_PULLUP);
pinMode(A1, INPUT_PULLUP);
pinMode(A2, INPUT_PULLUP);
motorInit();
lcd.begin(16, 2); // 16
col, 2
lignes
}

int
Button0(int
k)
{ // returns
the
state
of
the
button
static
int
state = OFF;
switch(state)
{
    case
OFF:
if (!digitalRead(k)) {state = PUSH;}
break;
case
PUSH: state = ON;
break;
case
ON:
if (digitalRead(k)) {state = RELEASE;}
break;
case
RELEASE: state = OFF;
break;
}
return state;
}

int
Button2(int
k)
{ // returns
the
state
of
the
button
static
int
state = OFF;
switch(state)
{
    case
OFF:
if (!digitalRead(k)) {state = PUSH;}
break;
case
PUSH: state = ON;
break;
case
ON:
if (digitalRead(k)) {state = RELEASE;}
break;
case
RELEASE: state = OFF;
break;
}
return state;
}

void
motorInit()
{
pinMode(11, OUTPUT);
pinMode(12, OUTPUT);
}

void
motorSet(int
command)
{
analogWrite(11, command);
digitalWrite(12, HIGH);
}

void
loop()
{
if (digitalRead(A1) == 1)
    {
        lcd.setCursor(0, 0); // Colonne
0, ligne
0
lcd.print("bonjour!");
lcd.setCursor(0, 1);
lcd.print(setpoint);
if (gestionBP1(A0) == PUSH)
{
    setpoint = setpoint * 0.9;
lcd.setCursor(0, 1);
lcd.print(setpoint);
lcd.print(" ");
}
if (gestionBP2(A2) == PUSH)
{
if (setpoint >= 10)
{setpoint=setpoint * 1.1;}
else {setpoint=setpoint+1;}
if (setpoint < 255)
{lcd.setCursor (0, 1);
lcd.print(setpoint);}
else {setpoint=255;
lcd.setCursor (0, 1);
lcd.print(setpoint);}
}
motorSet(setpoint);
}
else
{
if (millis() > deadline)
{setpoint_stop = setpoint_stop * 0.95;
analogWrite(11, setpoint_stop);
deadline += 100;}
if (setpoint_stop < 1)
{digitalWrite(A1, HIGH);
setpoint_stop=setpoint;
analogWrite(11, setpoint);}
}
}
