

const int flexPin = A7;
const int flexPin1 = A6;
const int flexPin2 = A5;
const int flexPin3 = A4;

void setup() {
    Serial.begin(9600);  // Start serial communication
}

void loop() {
    int value = analogRead(flexPin);
    int value1 = analogRead(flexPin1);
    int value2 = analogRead(flexPin2);
    int value3 = analogRead(flexPin3);

    // Map values from 0-1023 to 0-255
    value = map(value, 0, 1023, 0, 255);
    value1 = map(value1, 0, 1023, 0, 255);
    value2 = map(value2, 0, 1023, 0, 255);
    value3 = map(value3, 0, 1023, 0, 255);

    // Print sensor data as a CSV line
    Serial.print(value);
    Serial.print(" ");
    Serial.print(value1);
    Serial.print(" ");
    Serial.print(value2);
    Serial.print(" ");
    Serial.println(value3);  // Newline to end the row

    delay(1000);  // 1 second delay between readings
}
