The opencv function for SIFT has been removed in lastest versions of Opencv since its patented. 
Install the following python and opencv-python packages which has the SIFT functions to run the code in your system or create another virtual environment in conda and execute the same.

pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16

Versions:
python 3.6.3
opencv 3.4.2

OS - Windows

Code:
There are three python files namely,
Visual_odometry_final.py
Visual_odometry_Inbuilt_functions.py
Plot_view.py

Instruction to run Visual_odometry_final.py:
Place the Visual_odometry_final.py file inside the Oxford_dataset directory and run it. This will create a text file plotpoints.txt, which is called from the Plot_view.py file.
You can either choose to regenerate the plotpoints.txt or run the Plot_view.py file with the  plotpoints.txt file which has been provided along. 
Note: Regeneration of the plotpoints.txt may take upto 3 hours. Plotting using Plot_view.py with already generated plotpoints.txt textfile will take approximately 20 mins.
Make sure to uncomment line 17 and comment line 18 when running Plot_view.py

Instruction to run Visual_odometry_final.py:
Place the Visual_odometry_Inbuilt_functions.py file inside the Oxford_dataset directory and run it. This will create a text file plotpoints_inbuilt.txt, which is called from the Plot_view.py file.
You can either choose to regenerate the plotpoints_inbuilt.txt or run the Plot_view.py file with the  plotpoints_inbuilt.txt file which has been provided along. 
Note: Regeneration of the plotpoints_inbuilt.txt may take upto 3 hours. Plotting using Plot_view.py with already generated plotpoints_inbuilt.txt textfile will take approximately 20 mins.
Make sure to uncomment line 18 and comment line 17 when running Plot_view.py

Please do contact us if you have problems in running the code.

