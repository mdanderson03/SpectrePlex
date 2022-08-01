int ttl_pins[] = {6,7,8,9,10,11,12,13};

#define pump_sleep A7
#define pump_step A6
#define pump_dir A5

#define outer_chamber_pump_pwm 2
#define outer_chamber_pump_direction1 3
#define outer_chamber_pump_direction2 4 
#define drain_pump_pwm 5

#define IR_1 A0
#define IR_2 A2

int user_input;
int input_seq[3] = {0, 0, 0};
int i = 0;
int function = 0;
int funct_input_1 = 0;
int funct_input_2 = 0;

int calibration_steps = 50; //number steps to go from infrared sensor to fill chamber up. Must be calibrated!
int bubble_calibration_steps = 50; //number steps to go from tubing to just inside manifold single tube. Must be calibrated!
int fill_time = 10; // time in seconds to fill outer chamber up with PBS. Must be calibrated!

void setup() {
  // put your setup code here, to run once:

// activate TTL pins as outputs
for(int i=0; i<=7; i++){
  pinMode(ttl_pins[i], OUTPUT);
  
}

// activate pins for stepper peristaltic pump as outputs
pinMode(pump_sleep, OUTPUT);
pinMode(pump_step, OUTPUT);
pinMode(pump_dir,OUTPUT);

// activate pins for DC peristaltic pumps as outputs
pinMode(outer_chamber_pump_pwm, OUTPUT);
pinMode(outer_chamber_pump_direction1, OUTPUT);
pinMode(outer_chamber_pump_direction2, OUTPUT);
pinMode(drain_pump_pwm, OUTPUT);

// activate input pin for IR sensors
pinMode(IR_1, INPUT);
pinMode(IR_2, INPUT);

// put stepper to sleep
digitalWrite(pump_sleep, LOW);

// fix direction of PBS pump for outer chamber
digitalWrite(outer_chamber_pump_direction1, LOW);
digitalWrite(outer_chamber_pump_direction2, HIGH);

Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
//digitalWrite(A7, HIGH);
//step(100, pump_step, 2);
digitalWrite(13, HIGH);
  if (Serial.available()>0){
    i = 0;
    while (i <= 2){
   user_input= Serial.read();
   if (user_input != -1){
   input_seq[i] = user_input - 48;
   i++;
   }
    } 
    function = input_seq[0];
    funct_input_1 = input_seq[1];
    funct_input_2 = input_seq[2];   
  }

if (function == 1){
  chamber_fill_drain(funct_input_1);
  function = 0;
  funct_input_1 = 0;
  funct_input_2 = 0;
  Serial.println(0);
}

if (function == 2){
  bleach(funct_input_1, funct_input_2);
  function = 0;
  funct_input_1 = 0;
  funct_input_2 = 0;
  Serial.println(0);
}

if (function == 3){
  waiting(funct_input_1, funct_input_2);
  function = 0;
  funct_input_1 = 0;
  funct_input_2 = 0;
  Serial.println(0);
}

if (function == 4){
  wash(funct_input_1, funct_input_2);
  function = 0;
  funct_input_1 = 0;
  funct_input_2 = 0;
  Serial.println(0);
}

if (function == 5){
  stain(funct_input_1, funct_input_2);
  function = 0;
  funct_input_1 = 0;
  funct_input_2 = 0;
  Serial.println(0);
}

//end of loop!
}


//Functions are below

// fills and drains chamber
void chamber_fill_drain(int state){
  if( state == 1){
    digitalWrite(outer_chamber_pump_pwm, HIGH);
    delay(fill_time * 1000); // time in ms to fill chamber up
    digitalWrite(outer_chamber_pump_pwm, LOW);
  }
  if(state == 0){
    digitalWrite(drain_pump_pwm, HIGH);
    delay(fill_time * 1000 * 1.2); // time in ms to drain chamber
    digitalWrite(drain_pump_pwm, LOW);
  }
}

//chooses pinch valve and opens or closes it
void valve(int valve_number, int state){
if(state == 1){
  digitalWrite(ttl_pins[valve_number], HIGH);
}

if(state == 0){
  digitalWrite(ttl_pins[valve_number], LOW);
}
}

//wash for x amount of second at 2*y time gap between steps
void wash(int seconds, int step_time_gaps){
  int steps = (1000 * seconds) / (2 * step_time_gaps);
  valve(2,0); // un-pinch valve for PBS
  digitalWrite(pump_sleep, HIGH);
  step(steps, pump_step, step_time_gaps); // turn pump on for seconds amount of time
  digitalWrite(pump_sleep, LOW);
  valve(2,1); // pinch valve for PBS
}

// function to generate a bubble which is used to distinguish new fluid fronts
void bubble_insert(){
  valve(5,0); // open up air valve
  digitalWrite(pump_sleep, HIGH);
  step(bubble_calibration_steps, pump_step, 1); // move pump head to pull in air bubble. This number needs to be high enough to pull bubble just into main tube of mixing manifold
  valve(5,1); // close air valve
  digitalWrite(pump_sleep, LOW);
  }

//add bleach solution and let sit for yz amount of minutes
void bleach(int ten_minutes, int minutes){
  long bleach_time = ten_minutes * 60000 * 10 + minutes * 60000; // time that bleach solution will be on slide
  valve(3, 0); // un-pinch valve that connects to H202
  valve(4, 0); // un-pinch valve that connects to Base Bleach Stock
  bubble_insert(); //introduce bubble into system
  digitalWrite(pump_sleep, HIGH); // turn on stepper
  while(digitalRead(IR_1) != HIGH){ // listen to infrared fluid detector and wait until bubble hits it
    step(1, pump_step, 2);
  }
  while(digitalRead(IR_1) != LOW){ // list to infrared fluid detector and wait for bubble to pass and fluid is at it again
    step(1, pump_step, 1);
  }  
  step(calibration_steps,pump_step, 1); // go known number of steps to fill chamber up from sensor position
  digitalWrite(pump_sleep, LOW); // put stepper to sleep
  delay(bleach_time); // time bleach solution will be on slide
}

//put stain solutin onto slide with inc_time being in multiples of 10 minutes
void stain(int number, int inc_time){
  long wait = 60000 * 10 * inc_time; // convert inc_time into 10's of minutes in milli-seconds
  valve(number, 0); // un-pinch stain valve 
  bubble_insert();
  digitalWrite(pump_sleep, HIGH); // turn on stepper
  while(digitalRead(IR_1) != HIGH){ // listen to infrared fluid detector and wait until bubble hits it
    step(1, pump_step, 2);
  }
  while(digitalRead(IR_1) != LOW){ // list to infrared fluid detector and wait for bubble to pass and fluid is at it again
    step(1, pump_step, 1);
  }  
  step(calibration_steps,pump_step, 1); // go known number of steps to fill chamber up from sensor position
  digitalWrite(pump_sleep, LOW); // put stepper to sleep
  delay(wait);
}

// Just waits xy minutes
void waiting(int ten_minutes, int minutes){
  long wait = 60000 * (10* ten_minutes + minutes);
  delay(wait);
}

// Generic function to move stepper motor one step
void step(int step_count, int motor, int step_delay){
  for (i = 0; i <= step_count; i++){
    digitalWrite(motor, HIGH);
    delay(step_delay);
    digitalWrite(motor, LOW);
    delay(step_delay);
  }
}




  
