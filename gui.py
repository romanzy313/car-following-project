import math
import pygame
from gui.car import Car
from gui.constants import *


# from gui.draw import draw_road
# from gui.gui import draw_road

# Example file showing a circle moving on screen


def main():
    # pygame setup
    pygame.init()

    screen = pygame.display.set_mode(display_size)
    pygame.display.set_caption("Group 1 car follower project")
    clock = pygame.time.Clock()
    running = True
    dt = 0
    angle = 0
    speed = 100

    player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        screen.fill(display_color)
        pygame.draw.rect(screen, vis_color, vis_rect)
        pygame.draw.circle(screen, road_color, vis_center, road_radius, road_width)

        # Calculate the new position
        x = (
            vis_center[0]
            + road_radius * math.cos(math.radians(angle))
            - car_image_rect[2] / 2
        )
        y = (
            vis_center[1]
            + road_radius * math.sin(math.radians(angle))
            - car_image_rect[3] / 2
        )

        # screen.blit(rotate2d(car_image, (x,y), angle), (x,y))

        # rotate is not proper
        screen.blit(
            pygame.transform.rotate(car_image, -angle),
            (x + car_image_rect.centerx / 2, y + car_image_rect.centery / 2),
        )
        # pygame.draw.circle(screen, "red", player_pos, 40)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            speed += 100 * dt
        if keys[pygame.K_s]:
            speed -= 100 * dt

        angle += speed * dt

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000
    pygame.quit()


if __name__ == "__main__":
    main()
