#define heater_pin 5

char user_input;

void setup() {
  // put your setup code here, to run once:

// activate pins for DC peristaltic pumps as outputs
pinMode(heater_pin, OUTPUT);

// fix direction of PBS pump for outer chamber





digitalWrite(heater_pin, HIGH
);

Serial.begin(9600);
Serial.write(1);
}

void loop() {
  // put your main code here, to run repeatedly:

if (Serial.available()>0){
 user_input= Serial.read();
}


if (user_input == 'ON'){
  digitalWrite(heater_pin, HIGH);
  Serial.write('on');
}

if (user_input == 'OFF'){
  digitalWrite(heater_pin, LOW);
  Serial.write('off');
}

delay(500);

//end of loop!
}





  
