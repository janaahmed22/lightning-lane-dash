# Lightning Lane Dash

**Lightning Lane Dash** is a racing game controlled using hand gestures. An open palm is used to steer the car between lanes, while a peace sign activates the Kachow Boost.

The project was built using **Python, OpenCV, YOLO, and Pygame**.

## 1. Data Collection

We collected our own dataset of two hand gestures:

* Open palm 🖐️
* Peace sign ✌️

The images were taken using webcams under different lighting conditions, backgrounds, and hand positions. Data augmentation was also used to make the model more reliable.

## 2. Model Training

We trained a custom YOLO model using our collected dataset.

* **Model:** YOLOv8
* **Classes:** Open Palm, Peace Sign
* **Output:** `best.pt`

## 3. Game Implementation

The trained model was connected to the game through Pygame.

* 🖐️ **Open palm:** Controls the car's position between lanes.
* ✌️ **Peace sign:** Activates the Kachow Boost.
* **Obstacles:** Moving obstacles that the player needs to avoid.
* ⚡ **Nitro:** Can be collected and used for boosting.
* **Score:** Tracks the player's progress.

## 4. Testing

We tested the game with different users and under different lighting conditions to see how well the gesture detection worked during gameplay.

The game runs in real time, although detection can become less reliable in poor lighting or when the hand is not clearly visible.

## 5. Approach

The project is divided into two main parts:

* **Computer Vision:** OpenCV and YOLO are used to detect and classify the hand gestures.
* **Game:** Pygame handles the track, car movement, obstacles, nitro, and boost.

The YOLO output is then used as the input for the game controls.

## 6. Challenges

Some of the main challenges we faced were:

* Collecting enough varied images for the dataset.
* Getting reliable gesture detection in different lighting conditions.
* Connecting the YOLO detection results to the game in real time.
* Making the car movement and gesture controls feel responsive.

## 7. Results

The final game allows the player to control the car using hand gestures in real time. The project demonstrates how a custom-trained YOLO model can be connected to a simple game to create gesture-based controls.

## How to Run

Clone the repository:

```bash
git clone https://github.com/janaahmed22/lightning-lane-dash.git
cd lightning-lane-dash
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run the game:

```bash
python main.py
```
