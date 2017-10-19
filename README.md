# Machine-learning

Contains solutions to machine-learning problems hosted on various sites like Hackerearth

### RoadSign Prediction : 

Predict which direction a road sign is applicable to given the following features :

1. **ID** - a unique identifier for each record in dataset.
2. **DetectedCamera** - imagine a car, which is fitted with 4 cameras on top, one each facing front, right, rear and left. Each cameras clicks pictures on every few meters as car moves. DetectedCamera value tells you on which camera image a road sign was observed/found, by an image detection software.
3. **AngleOfSign**	- values are in degrees ranging from 0 to 360 in clockwise direction from the front of car, indicates the angle from the front of the car to the direction in which the sign is detected.
4. **SignWidth**	- width of the sign bounding box in the image in pixels.
5. **SignHeight** - height of the sign bounding box in the image in pixels.
6. **SignAspectRatio** -	this is the width/height ratio of the sign bounding box, derived from SignWidth/SignHeight. Can provide an indication that sign is facing camera or not. A sign facing the driver, detected on an image captured from almost 80 degrees from front (on right camera), will have a bounding box that is skewed from its original aspect ratio. If its facing the right camera, it will have nearly original aspect ratio of the sign.
7. **SignFacing (Target)** -	For the above inputs, where the sign is actually facing is captured here, from manually reviewed sign facing records.

I finished under top 15 in this competetion.
There was a dataleak i missed. Row ids closer to each other usually belonged to same class.

### Predict the Segment - Hotstar :

classify customers based on watch patterns, learn patterns from customers whose watch patterns are already known. 

1. **ID** -	unique identifier variable.
2. **titles** -	titles of the shows watched by the user and watch_time on different titles in the format “title:watch_time” separated by comma, e.g. “JOLLY LLB:23, Ishqbaaz:40”. watch_time is in seconds
3. **genres** -	same format as titles.
4. **cities** -	same format as titles.
5. **tod** -	total watch time of the user spreaded across different time of days (24 hours format) in the format “time_of_day:watch_time” separated by comma, e.g. “1:454, “17”:5444”.
6. **dow** -	total watch time of the user spreaded across different days of week (7 days format) in the format “day_of_week:watch_time” separated by comma, e.g. “1:454, “6”:5444”.
7. **segment** - target variable. consider them as interest segments. For modeling, encode pos = 1, neg = 0.

Finished in the top 10% and qualified to the second round after combining both road sign and segment prediction scores.
Using an lgbm model would got the highest score on the leader board.

### Predict Ad-clicks :

Predict the probability whether an ad will get clicked or not.
 
1. **ID** - Unique ID
2. **datetime** - timestamp
3. **siteid** - website id
4. **offerid** -	offer id (commission based offers)
5. **category** -	offer category
6. **merchant** - seller ID
7. **countrycode** -	country where affiliates reach is present
8. **browserid** -	browser used
9. **devid** -	device used
10. **click**	- target variable

### Predict Happiness based on Hotel reviews :
Text mininng to predict customer happiness.

### Whats cooking ?
Predict cuisine based on ingredients.



### 
