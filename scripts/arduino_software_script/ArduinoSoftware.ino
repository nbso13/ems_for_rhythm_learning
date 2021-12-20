/**
 * ArduinoSoftware_Arduino_IDE
 *
 *  Copyright 2016 by Tim Duente <tim.duente@hci.uni-hannover.de>
 *  Copyright 2016 by Max Pfeiffer <max.pfeiffer@hci.uni-hannover.de>
 *
 *  Licensed under "The MIT License (MIT) - military use of this product is forbidden - V 0.2".
 *  Some rights reserved. See LICENSE.
 *
 * @license "The MIT License (MIT) - military use of this product is forbidden - V 0.2"
 * <https://bitbucket.org/MaxPfeiffer/letyourbodymove/wiki/Home/License>
 */

/**
 * Minor Revisions to this file by Pedro Lopes <plopesresearch@gmail.com>, all credit remains with the original authors (see above).
 */

// Necessary files (AltSoftSerial.h, AD5252.h, Rn4020BTLe.h, EMSSystem.h, EMSChannel.h) and dependencies (Wire.h, Arduino.h)
#include "ArduinoSoftware.h"
#include "AltSoftSerial.h"
#include "Wire.h"
#include "AD5252.h"
#include "Rn4020BTLe.h"
#include "EMSSystem.h"
#include "EMSChannel.h"
#include "Debug.h"

//the string below is how your EMS module will show up for other BLE devices
#define EMS_BLUETOOTH_ID "EMSSEM43"

//setup for accepting commands also via USB (accepts USB commands if ACCEPT_USB_COMMANDS is 1)
#define ACCEPT_USB_COMMANDS 1

//setup the Bluetooth module. This is necessary if the toolkit is programmed the first time or if Bluetooth parameter are changed
#define SETUP_BLUETOOTH 0

//Initialization of control objects
AltSoftSerial softSerial;
AD5252 digitalPot(0);
Rn4020BTLe bluetoothModule(2, &softSerial);
EMSChannel emsChannel1(5, 4, A2, &digitalPot, 1);
EMSChannel emsChannel2(6, 7, A3, &digitalPot, 3);
EMSSystem emsSystem(2);

String command = "";
void setup() {
	Serial.begin(115200);

	softSerial.setTimeout(100);
	debug_println(F("\nSETUP:"));

	if (SETUP_BLUETOOTH) {
		//Reset and initialize the Bluetooth module
		debug_println(F("\tBT: RESETTING"));
		bluetoothModule.reset();
		debug_println(F("\tBT: RESET DONE"));
		debug_println(F("\tBT: INITIALIZING"));
		bluetoothModule.init(EMS_BLUETOOTH_ID);
	} else {
		//Or only start the communication if the Bluetoothmodule is setup correctly.
		bluetoothModule.start_communication();
	}

	debug_println(F("\tBT: INITIALIZED"));

	//Add the EMS channels and start the control
	debug_println(F("\tEMS: INITIALIZING CHANNELS"));
	emsSystem.addChannelToSystem(&emsChannel1);
	emsSystem.addChannelToSystem(&emsChannel2);
	EMSSystem::start();
	debug_println(F("\tEMS: INITIALIZED"));
	debug_println(F("\tEMS: STARTED"));
	debug_println(F("SETUP DONE (LED 13 WILL BE ON)"));

	emsSystem.shutDown();

	command.reserve(21);

	pinMode(13, OUTPUT);
	digitalWrite(13, HIGH);
}

String hexCommandString;
const String BTLE_DISCONNECT = "Connection End";

void loop() {
	if (softSerial.available() > 0) {
		String notification = softSerial.readStringUntil('\n');
		notification.trim();
		debug_println(F("\tBT: received message: "));
		debug_println(notification);

		//Data messages from the RN4020 start with WN and look like the following WN,001C,1ABD3467. First 8 character are not relevant. But for offering more services the second parameter is the service ID
		if (notification.length() >= 2 && notification.charAt(0) == 'W'
				&& notification.charAt(1) == 'V') {

			for (uint8_t i = 8; i < notification.length() - 1; i = i + 2) {
				int nextChar = convertTwoHexCharsToOneByte(notification, i);
				if (nextChar == -1) {
					debug_println(F("Failed to convert"));
				} else {
					command = command + (char) nextChar;
				}
			}

//			if(command.charAt(0) != 'x') { // if string does not begin with x, this is a debug format str
//          debug_println(F("\tEMS_CMD: 1 Converted command: "));
//          debug_println(command);
//          debug_Toolkit(command);
//        } 
//       else { //then this is properly formatted command
       String command_str = command.substring(1); //cut the x for formatting
       debug_println(F("\tEMS_CMD: 2 Converted command: "));
       debug_println(command_str);
       emsSystem.doCommand(&command_str);
//      }

			command = "";
		} else if (notification.equals(BTLE_DISCONNECT)) {
			debug_println(F("\tBT: Disconnected"));
			emsSystem.shutDown();
		}

	}

	//Checks whether a signal has to be stoped
	if (emsSystem.check() > 0) {

	}

   if (ACCEPT_USB_COMMANDS) {
      if (Serial.available() > 0) {
        
        debug_println("received!"); 
//        float time1 = millis();
        String str = Serial.readStringUntil('\n');
//        Serial.write("received! 2 \n");
//        debug_println(str); 
//        Serial.write("received! 3 \n");
//        if(str.charAt(0) != 'x') { // if string does not begin with x, this is a debug format str
//          debug_Toolkit(str);
//        } else { //then this is properly formatted command
//         Serial.write("received! 4 \n");
  			 String command = str; //cut the x for formatting
//         debug_println(F("\tEMS_CMD: Converted command: "));
//         debug_println(command);

//         float time2 = millis();
          debug_println("act_command");
  			  emsSystem.doCommand(&command);
          debug_println("done");
//         float time_between = time2-time1;
//         char buff[10];
//         debug_println(itoa(time_between, buff, 10));
  
//  		}
  	}
  
  }
}

//Convert-functions for HEX-Strings "4D"->"M"
int convertTwoHexCharsToOneByte(String &s, uint8_t index) {
	int byteOne = convertHexCharToByte(s.charAt(index));
	int byteTwo = convertHexCharToByte(s.charAt(index + 1));
	if (byteOne != -1 && byteTwo != -1)
		return (byteOne << 4) + byteTwo;
	else {
		return -1;
	}
}

int convertHexCharToByte(char hexChar) {
	if (hexChar >= 'A' && hexChar <= 'F') {
		return hexChar - 'A' + 10;
	} else if (hexChar >= '0' && hexChar <= '9') {
		return hexChar - '0';
	} else {
		return -1;
	}
}

//For testing
void debug_Toolkit(String str) {
  if (str.length() > 1) { // should be m100l5 = metronome program, 100 bpm, for 5 seconds
    if (str.charAt(0) == 'm') {// 'metronome' on program
      int len_index = str.indexOf("l");
      String bpm_substr = str.substring(1, len_index); //bpm
      String len_substr = str.substring(len_index+1);
      int bpm = bpm_substr.toInt();
      int length_of_stim = len_substr.toInt(); //seconds
      
      if (bpm < 1) {
         debug_println("bpm incorrectly formatted");
         return;
      }
      
      debug_println(String("bpm is ") + bpm_substr + String(", stim len is ") + len_substr);
      metronome(bpm, length_of_stim);
    }
    else if (str.charAt(0) == 'r') { //rhythm program: should be r10001000l4t100 = 1 measure rhythm, rhythm in 8 subdivisions,
      // length of repeat = 4, tempo is 100 bpm
      int len_index = str.indexOf("l");
      int tempo_index = str.indexOf("t");
      String rhythm_substr = str.substring(1, len_index); //pattern
      String len_substr = str.substring(len_index+1, tempo_index); //repeats
      String tempo_substr = str.substring(tempo_index+1); // tempo
      int repeats = len_substr.toInt(); 
      int bpm = tempo_substr.toInt(); 
      
      debug_println(String("Pattern is ") + rhythm_substr + String(", repeat num is ") + len_substr + String(", bpm is ") + tempo_substr);
      play_rhythm(rhythm_substr, repeats, bpm);
      
    }
  } else {
    char c = str.charAt(0);
  	if (c == '1') {
  		if (emsChannel1.isActivated()) {
  			emsChannel1.deactivate();
  			debug_println(F("\tEMS: Channel 1 inactive"));
  		} else {
  			emsChannel1.activate();
  			debug_println(F("\tEMS: Channel 1 active"));
  		}
  	} else if (c == '2') {
  		if (emsChannel2.isActivated()) {
  			emsChannel2.deactivate();
  			debug_println(F("\tEMS: Channel 2 inactive"));
  		} else {
  			emsChannel2.activate();
  			debug_println(F("\tEMS: Channel 2 active"));
  		}
  	} else if (c == 'q') {
  		digitalPot.setPosition(1, digitalPot.getPosition(1) + 1);
  		debug_println(
  				String(F("\tEMS: Intensity Channel 1: "))
  						+ String(digitalPot.getPosition(1)));
  	} else if (c == 'w') {
  		digitalPot.setPosition(1, digitalPot.getPosition(1) - 1);
  		debug_println(
  				String(F("\tEMS: Intensity Channel 1: "))
  						+ String(digitalPot.getPosition(1)));
  	} else if (c == 'e') {
  		//Note that this is channel 3 on Digipot but EMS channel 2
  		digitalPot.setPosition(3, digitalPot.getPosition(3) + 1);
  		debug_println(
  				String(F("\tEMS: Intensity Channel 2: "))
  						+ String(digitalPot.getPosition(3)));
  	} else if (c == 'r') {
  		//Note that this is channel 3 on Digipot but EMS channel 2
  		digitalPot.setPosition(3, digitalPot.getPosition(3) - 1);
  		debug_println(
  				String(F("\tEMS: Intensity Channel 2: "))
  						+ String(digitalPot.getPosition(3)));
  	}
  }
}



void metronome(int bpm, int length_of_stim) { //in beats per minute and lengthofstim is seconds
  //max bpm is __ as given actual stim length of ___ms as you can only fit around 6 stims in a 
  // second.
  
  int actual_stim_length = 300; //ms
  double max_bpm = 60*floor(1000/actual_stim_length);
  
  if (bpm > max_bpm) {
    debug_println(String("max metronome bpm is " + String(max_bpm)));
    return;
  }

  // determine wait time between pulses
  double milliseconds_per_beat = 60000/bpm;
  double milliseconds_wait = milliseconds_per_beat - actual_stim_length;
  int total_repeats = floor(length_of_stim*1000/milliseconds_wait); // det
  for (uint8_t i = 0; i < total_repeats; i = i + 1) {
    digitalPot.setPosition(3, 0);
    debug_println(
          String(F("\tEMS: Intensity Channel 2: "))
              + String(digitalPot.getPosition(3)));
    delay(actual_stim_length);
    digitalPot.setPosition(3, 255);
    debug_println(
        String(F("\tEMS: Intensity Channel 2: "))
            + String(digitalPot.getPosition(3)));
    delay(milliseconds_wait);
    
  }
}

void play_rhythm(String rhythm_substr, int repeats, int bpm) {
  
  int actual_stim_length = 150; //ms
  double max_bpm = floor(30000/actual_stim_length); // how many eighthnote pulses could you fit into a minute without overlapping?
  
  if (bpm > max_bpm) {
    debug_println(String("max metronome bpm is " + String(max_bpm)));
    return;
  }

  // determine pulse+wait length
  double milliseconds_per_eighthnote = 30000/bpm;
  double milliseconds_wait = milliseconds_per_eighthnote - actual_stim_length;
  
  for (uint8_t i = 0; i < repeats; i = i + 1) { // present the rhythm with appropriate number of repeats
    for (uint8_t j = 0; j < rhythm_substr.length(); j = j + 1) { // go through each eighthnote in the pattern
      if (rhythm_substr.charAt(j) == '1') { //note
        digitalPot.setPosition(3, 0);
        debug_println(
              String(F("\tEMS: Intensity Channel 2: "))
                  + String(digitalPot.getPosition(3)));
        delay(actual_stim_length);
        digitalPot.setPosition(3, 255);
        debug_println(
            String(F("\tEMS: Intensity Channel 2: "))
                + String(digitalPot.getPosition(3)));
        delay(milliseconds_wait);
      } else if(rhythm_substr.charAt(j) == '0') { // rest
        delay(milliseconds_per_eighthnote);
      } else { 
        debug_println(String("malformed rhythm pattern: ") + rhythm_substr);
        break;
      }
    }
  }
    
}
