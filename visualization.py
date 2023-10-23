# import math
import pygame
import pygame_gui
from visualization.car import Car
from visualization.constants import *

import math
import json


def rot_center(image, angle, x, y):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(center=(x, y)).center)
    return rotated_image, new_rect


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
    pygame.draw.rect(
        surf, (255, 0, 0), (*rotated_image_rect.topleft, *rotated_image.get_size()), 2
    )


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


def main():
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode(display_size)
    manager = pygame_gui.UIManager(display_size)

    hello_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((350, 275), (100, 50)),
        text="Say Hello",
        manager=manager,
    )

    pygame.display.set_caption("Car follower visualization")
    clock = pygame.time.Clock()
    running = True
    dt = 0
    angle = 0
    speed = 100

    result = load_result("results/test_run.json")

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == hello_button:
                    print("Hello World!")

            manager.process_events(event)

        manager.update(dt)

        # fill the screen with a color to wipe away anything from last frame
        screen.fill(display_color)
        pygame.draw.rect(screen, vis_color, vis_rect)
        pygame.draw.circle(screen, road_color, vis_center, road_radius, road_width)

        # x = vis_center[0] + 30
        # y = vis_center[1]

        # screen.blit(rotate2d(car_image, (x,y), angle), (x,y))

        # rotate is not proper
        # screen.blit(
        #     pygame.transform.rotate(car_image, -angle),
        #     (x + car_image_rect.centerx / 2, y + car_image_rect.centery / 2),
        # )
        # pygame.draw.circle(screen, "red", player_pos, 40)

        drawVehicle(screen, car_image, angle)
        drawVehicle(screen, car_image, angle + 90)
        drawVehicle(screen, car_image, angle + 180)
        drawVehicle(screen, car_image, angle + 270)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            speed += 100 * dt
        if keys[pygame.K_s]:
            speed -= 100 * dt

        angle += speed * dt

        manager.draw_ui(screen)
        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000
    pygame.quit()


if __name__ == "__main__":
    main()
