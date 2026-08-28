import cv2
import pygame
import random
import time
from ultralytics import YOLO

#basic game settings
SCREEN_W, SCREEN_H = 800, 600
LANES = 3         
START_LIVES = 3
BOOST_DURATION = 1.5

#falling stuff (obstacles or nitro cans)
class FallingThing:
    def __init__(self, lane, kind, speed):
        self.lane = lane
        self.kind = kind
        self.y = -20   #start above screen
        self.speed = speed

    def move_down(self, dt):
        self.y += self.speed * dt

#main game state
class RaceGame:
    def __init__(self):
        self.car_lane = 1
        self.score = 0
        self.nitro_points = 0
        self.lives = START_LIVES
        self.boost_end = 0
        self.objects = []
        self.start_time = time.time()
        self.best_score = 0
        self.game_over = False

    #move car based on hand x position
    def steer_car(self, hand_x, cam_w):
        lane = int(hand_x / cam_w * LANES)
        self.car_lane = max(0, min(LANES-1, lane))

    #activate boost if nitro available
    def use_boost(self, now):
        if self.nitro_points > 0 and now >= self.boost_end:
            self.nitro_points -= 1
            self.boost_end = now + BOOST_DURATION
            #clear obstacles in current lane
            self.objects = [o for o in self.objects if not (o.kind=="obstacle" and o.lane==self.car_lane)]

    #update falling objects and check collisions
    def update_objects(self, dt, now):
        new_list = []
        for o in self.objects:
            o.move_down(dt)
            #check collision with car
            if (o.y-(SCREEN_H-80))<40 and o.lane==self.car_lane:
                if o.kind=="nitro":
                    self.nitro_points += 1
                    self.score += 10
                elif now>=self.boost_end:
                    self.lives -= 1
                    self.score = max(0,self.score-20)
                    if self.lives<=0:
                        self.game_over = True
                continue
            if o.y<SCREEN_H+50: new_list.append(o)
        self.objects = new_list

    #draw car
    def draw_car(self, surf, lane, y, color):
        lane_w = SCREEN_W//LANES
        x = lane_w*lane + lane_w//2
        pygame.draw.rect(surf, color, (x-20,y-40,40,60))

    #draw nitro or obstacle
    def draw_object(self, surf, obj):
        lane_w = SCREEN_W//LANES
        x = lane_w*obj.lane + lane_w//2
        if obj.kind=="nitro":
            pygame.draw.circle(surf,(0,200,255),(x,int(obj.y)),15)
        else:
            pygame.draw.circle(surf,(80,80,80),(x,int(obj.y)),20)

#detect gestures
def detect_gesture(model, frame):
    hand_x = None; peace_sign = False
    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        label = results.names[int(box.cls[0])].lower()
        conf = float(box.conf[0])
        if conf<0.25: continue
        if "peace" in label: peace_sign = True
        if "palm" in label:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            hand_x = (x1+x2)/2
    return hand_x, peace_sign

def main():
    model = YOLO("best.pt")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Camera not found")

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W,SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None,30)
    game = RaceGame()
    running = True
    peace_prev = False

    while running:
        now = time.time()
        dt = clock.get_time()/1000

        #handle keyboard events
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_r and game.game_over: game = RaceGame()
                elif e.key==pygame.K_LEFT: game.car_lane = max(0,game.car_lane-1)
                elif e.key==pygame.K_RIGHT: game.car_lane = min(LANES-1,game.car_lane+1)
                elif e.key==pygame.K_SPACE: game.use_boost(now)

        #camera + gesture detection
        ok, frame = cam.read()
        if ok and not game.game_over:
            hand_x, peace = detect_gesture(model, frame)
            if hand_x: game.steer_car(hand_x, frame.shape[1])
            if peace and not peace_prev: game.use_boost(now)
            peace_prev = peace

        #difficulty scaling
        if not game.game_over:
            elapsed = now-game.start_time
            speed = 150+min(elapsed*5,200)
            spawn_rate = max(0.3,0.8-elapsed*0.01)
            if random.random()<dt/spawn_rate:
                kind = "nitro" if random.random()<0.25 else "obstacle"
                game.objects.append(FallingThing(random.randrange(LANES),kind,speed))
            game.update_objects(dt,now)

        #draw everything
        screen.fill((30,30,30))
        #lane lines
        for lane in range(1,LANES):
            x = SCREEN_W*lane//LANES
            pygame.draw.line(screen,(200,200,200),(x,65),(x,SCREEN_H),2)

        for o in game.objects: game.draw_object(screen,o)
        car_color = (255,0,0) if now>game.boost_end else (255,200,0)
        game.draw_car(screen,game.car_lane,SCREEN_H-80,car_color)
        hud = font.render(f"Score {game.score} Lives {game.lives} Nitro {game.nitro_points}",True,(255,255,255))
        screen.blit(hud,(20,20))

        #camera preview window
        if ok:
            small = cv2.resize(frame,(200,120))
            small = cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
            cam_surf = pygame.surfarray.make_surface(small.swapaxes(0,1))
            screen.blit(cam_surf,(SCREEN_W-210,20))

        if game.game_over:
            msg = font.render("GAME OVER - R to restart",True,(255,255,255))
            screen.blit(msg,(SCREEN_W//2-100,SCREEN_H//2))
        pygame.display.flip()
        clock.tick(60)

    cam.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__=="__main__":
    main()


