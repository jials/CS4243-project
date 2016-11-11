Team Number: 17
Team Member:
Chua Chin Siang A0112089J
Lee Kim Hua @ Michael Lee A0112139R
Cheong Yuan Xiang A0112162Y
Choo Jia Le A0116673A

********************************************
***** BeachVolleyBall Tracking Program *****
********************************************

This program aims to process the seven beach volleyball match videos and output respective video which contains original video, panorama video, topdown view of the match and statistics table. 

Below are the available operations for the user to execute:
*beachVolleyball1.mov is used as the sample filename for below explanation purposes.

------------
| handpick | * The skip frames is set to 20
------------
This command allows user to handpick up to seven interest points of every 20th frame of the video, after the handpicking is done, a panorama video will be generated.
- Command to execute: python project.py -o handpick -f beachVolleyball1.mov 

* Optional parameter: -s

User may include the optional parameter to indicate the starting frame index of the pickle file and onwards to be repicked, once it is done, a panorama video will be generated.
- Command to execute: python project.py -o handpick -f beachVolleyball1.mov -s 5

User may include the optional parameter with the value of -1 to generate the panorama video using an existing pickle file, this command is designed for user convenience.
- Command to execute: python project.py -o handpick -f beachVolleyball1.mov -s -1

-----------
| topdown |
-----------
This command allows user to first indicate the four courners of the panorama video to be mapped to the four courners of the beach volleyball court top down view image. Then the user goes through every 20th frame to handpick the position of the four players and the ball. Besides, user also able to indicate if any of the players is jumping, and which player is currently tapping the ball. After the handpicking is done, a topdown view video and the statistic table video will be generated. Currently we are interested in the distance travelled and the number of jumps by each player in a video.
- Command to execute: python project.py -o topdown -f beachVolleyball1.avi

* Optional parameter: -s

User may include the optional parameter with the value of -1 to generate the topdown view video and the statistic table video using existing players, ball and jump pickle files, this command is designed for user convenience.
- Command to execute: python project.py -o topdown -f beachVolleyball1.avi -s -1

----------
| stitch |
----------
* Compulsory parameter: -d

This command allows user to stitch two videos that have different camera position and/or angle into one video with the same camera position and angle, the parameter -d is to indicate which camera angle and position to follow, if 1 is indicated, then the second video will be stitched to the first video with first video camera position and angle.
- Command to execute: python project.py -o stitch -f beachVolleyball1_first.avi,beachVolleyball1_second.avi -d 1

-------
| cut |
-------
* Compulsory parameter: -t

This command allows user to cut video at a specific timing in seconds.
- Command to execute: python project.py -o cut -f beachVolleyball1.mov -t 3

---------------
| concatenate |
---------------
This command allows user to concatenate four videos into one.
- Command to execute: 
python project.py -o concatenate beachVolleyball1.mov,beachVolleyball1_panorama.avi,beachVolleyball1_topdown.avi,beachVolleyball1_stats.avi

=============================================================================

Below gives a brief explanation to our main source code:
--------------
| project.py |
--------------
It is the main executor of our program, it will parse the user input command and execute relevant operation.

----------------------
| changeDetection.py |
----------------------
It performs the Lucas Kanade change detection method. 

-----------------
| statistics.py |
-----------------
It handles all statistics relevant requests. 

------------------
| imageMarker.py |
------------------
It provides image marking functions.

--------------------
| handpickPixel.py |
--------------------
It handles different image marking request.

-----------
| util.py |
-----------
It stores all the helper functions.

=============================================================================

Below shows some of the experienced codes and methods but are currently not in used:
----------------------
| cornerDetection.py | - Detect corners of all frames of the video
----------------------
--------------------
| edgeDetection.py | - Detect edges of all frames of the video
--------------------
------------------
| convolution.py | - Served as helper class for corner and edge detections
------------------
