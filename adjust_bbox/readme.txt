This code is used to manually drawing the bounding box. 

Occasionally, the pre-defined threshold cannot detect the gloves correctly. Under the circumstances, 
I use this program to manually draw the bounding boxes.

Instructions:
1. When running the program, two windows will pop up. 
	- On "Rectangle Window (Esc: escape)", draw the bounding box manually using the mouse. After finishing the bounding box, press 'Esc'.
	- The new bounding box will show in "Result (f: forward)(q: quit)" window. If you are satisfied with the result, press the 'f' key to 
	  move to the next frame. Otherwise, press the space bar to allow you to re-draw the bounding box. In comparison, the 'q' key is for 
	  exiting the program but without updating the original bounding box coordinates.