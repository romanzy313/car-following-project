import pygame
from src.read_data import read_data


class Vehicle:
    x: float
    v: float
    a: float

    def __init__(self):
        # body of the constructor
        return

    def tick(self, dt, newAcceleration):
        self.a = newAcceleration
        self.v = self.v + newAcceleration * dt
        self.x = self.x + self.v * dt

        return


def main():
    # load zaar data
    data = read_data(cfpair="AH", dataset="train")

    print("data is", data)

    screen_width = 1280
    screen_height = 720

    road_radius = 720 / 2 - 50

    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    running = True
    dt = 0

    player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

    while running:
        # screen =
        center = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("purple")

        pygame.draw.circle(screen, "red", player_pos, 40)

        pygame.draw.circle(screen, "blue", center, road_radius, 20)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            player_pos.y -= 300 * dt
        if keys[pygame.K_s]:
            player_pos.y += 300 * dt
        if keys[pygame.K_a]:
            player_pos.x -= 300 * dt
        if keys[pygame.K_d]:
            player_pos.x += 300 * dt

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000
    pygame.quit()


if __name__ == "__main__":
    main()
