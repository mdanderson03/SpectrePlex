#define heater_pin 2
#define heater_current_dir 3

char user_input;

void setup() {
  // put your setup code here, to run once:

// activate pins for DC peristaltic pumps as outputs
pinMode(heater_pin, OUTPUT);
pinMode(heater_current_dir, OUTPUT);

// fix direction of PBS pump for outer chamber
digitalWrite(heater_current_dir, LOW);
digitalWrite(heater_pin, LOW);

Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:

if (Serial.available()>0){
 user_input= Serial.read();
}


if (user_input == 'ON'){
  digitalWrite(heater_pin, HIGH);
}

if (user_input == 'OFF'){
  digitalWrite(heater_pin, LOW);
}

delay(500);

//end of loop!
}





  
