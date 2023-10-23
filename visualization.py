# import math
from typing import Any, List
import pygame
import pygame_gui
from visualization.car import Car
from visualization.constants import *

import math
import json


def blitRotate(surf, image, pos, originPos, angle):
    # offset from pivot to center
    image_rect = image.get_rect(topleft=(pos[0] - originPos[0], pos[1] - originPos[1]))
    offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center

    # roatated offset from pivot to center
    rotated_offset = offset_center_to_pivot.rotate(-angle)

    # roatetd image center
    rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_image_rect = rotated_image.get_rect(center=rotated_image_center)

    # rotate and blit the image
    surf.blit(rotated_image, rotated_image_rect)

    # draw rectangle around the image
    # pygame.draw.rect(
    #     surf, (0, 0, 255), (*rotated_image_rect.topleft, *rotated_image.get_size()), 2
    # )


def drawVehicle(surf, image, angle):
    # Calculate the new position
    w, h = car_image.get_size()

    x = vis_center[0] + (road_radius - road_width / 2) * math.sin(math.radians(angle))
    y = vis_center[1] + (road_radius - road_width / 2) * math.cos(math.radians(angle))
    blitRotate(surf, image, (x, y), (w / 2, h / 2), angle - 90)

    pass


def load_result(name: str):
    with open(name) as f:
        data = json.load(f)
        return data
        # print(d)


result = load_result("results/test_run.json")
road_length: float = result["scene"]["road_length"]
steps = result["steps"]
iteration_count: int = len(steps) - 1
outcome: str = result["outcome"]

iteration: int = 0


def next_iteration():
    global iteration
    if iteration == iteration_count:
        iteration = 0
    else:
        iteration += 1


# pygame
pygame.init()
screen = pygame.display.set_mode(display_size)
pygame.display.set_caption("Car follower visualization")
pygame.font.init()
my_font = pygame.font.SysFont("Ubuntu", 18)
clock = pygame.time.Clock()

manager = pygame_gui.UIManager(display_size)

running = True
playing = True
dt = 0
angle = 0
speed = 100


def render_text(surf, text: str, position, color="white"):
    text_surface = my_font.render(text, True, color)
    surf.blit(text_surface, position)


def toggle_playing():
    global playing
    playing = not playing
    start_stop_button.set_text("Stop" if playing else "Start")
    print("playing is", playing)


start_stop_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(start_stop_rect),
    text="Stop",
    manager=manager,
)

frame_bar = pygame_gui.elements.UIHorizontalScrollBar(
    manager=manager,
    relative_rect=pygame.Rect(progress_rect),
    visible_percentage=0.1,
)
frame_bar.enable_arrow_buttons = True


while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                toggle_playing()

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == start_stop_button:
                toggle_playing()
                # frame_bar.reset_scroll_position()

        manager.process_events(event)

    manager.update(dt)

    # draw the basics
    screen.fill(display_color)
    pygame.draw.rect(screen, vis_color, vis_rect)
    pygame.draw.circle(screen, road_color, vis_center, road_radius, road_width)

    # draw vehicles
    vehicles = steps[iteration]["vehicles"]
    for vehicle in vehicles:
        angle = (vehicle["position"] / road_length) * 360
        drawVehicle(screen, car_image, angle)

    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_SPACE]:
    #     print("spacebaar")
    #     playing = not playing
    # if keys[pygame.K_s]:
    #     speed -= 100 * dt

    if playing:
        next_iteration()

    # draw ui
    render_text(screen, f"Iteration: {iteration}/{iteration_count}", iteration_rect)
    render_text(screen, f"Outcome: {outcome}", outcome_rect)

    manager.draw_ui(screen)
    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000
pygame.quit()
