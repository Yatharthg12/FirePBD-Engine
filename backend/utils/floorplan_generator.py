import cv2
import numpy as np
import random

IMG_SIZE = 512
WALL_THICKNESS = 6
DOOR_WIDTH = 20

def split_space(x, y, w, h, depth=0):
    if depth > 3 or w < 100 or h < 100:
        return [(x, y, w, h)]

    if random.random() > 0.5:
        # vertical split
        split = random.randint(int(w*0.3), int(w*0.7))
        return split_space(x, y, split, h, depth+1) + \
               split_space(x+split, y, w-split, h, depth+1)
    else:
        # horizontal split
        split = random.randint(int(h*0.3), int(h*0.7))
        return split_space(x, y, w, split, depth+1) + \
               split_space(x, y+split, w, h-split, depth+1)


def draw_walls(img, rooms):
    for (x, y, w, h) in rooms:
        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (0,),
            WALL_THICKNESS
        )


def add_doors(img, rooms):
    for i in range(len(rooms)-1):
        (x1, y1, w1, h1) = rooms[i]
        (x2, y2, w2, h2) = rooms[i+1]

        # find shared boundary
        if abs((x1+w1) - x2) < 10:
            # vertical wall → door gap
            dy = random.randint(y1+20, y1+h1-20)
            cv2.line(img,
                     (x1+w1, dy),
                     (x1+w1, dy+DOOR_WIDTH),
                     (255,),
                     WALL_THICKNESS)

        if abs((y1+h1) - y2) < 10:
            dx = random.randint(x1+20, x1+w1-20)
            cv2.line(img,
                     (dx, y1+h1),
                     (dx+DOOR_WIDTH, y1+h1),
                     (255,),
                     WALL_THICKNESS)


def generate_floorplan():
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255

    margin = 20
    rooms = split_space(margin, margin, IMG_SIZE-2*margin, IMG_SIZE-2*margin)

    draw_walls(img, rooms)
    add_doors(img, rooms)

    return img


if __name__ == "__main__":
    for i in range(10):
        img = generate_floorplan()
        cv2.imwrite(f"real_fp_{i}.png", img)